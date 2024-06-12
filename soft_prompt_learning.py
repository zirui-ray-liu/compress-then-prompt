from tqdm import tqdm
from itertools import chain
import torch
import argparse
import numpy as np

import time
import torch.nn.functional as F
import os
import ipdb
from collections.abc import Mapping
from transformers import AutoTokenizer, set_seed, default_data_collator
from datasets import load_dataset
from typing import Any, Union
from prompt import LLamaPromptTuningLM, OPTPromptTuningLM, llama_loader, TextDataset, BloomPromptTuningLM
from datasets import Dataset
from accelerate import Accelerator
# from optimum.bettertransformer import BetterTransformer


parser = argparse.ArgumentParser("")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--dataset",type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--project_root", default=ROOT, help="Root to save the results and logs")
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--init_from_vocab", action="store_false")
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--dataloader_num_workers", type=int, default=16)
parser.add_argument("--dataloader_pin_memory", action="store_true")
parser.add_argument("--seqlen", type=int, default=1024)
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)


def freeze_model(model):
    for n, m in model.named_parameters():
        if "soft_prompt" in n:
            m.requires_grad = True 
        else:
            m.requires_grad = False

    tot_params = sum(p.numel() for p in model.parameters())
    print("***** Model Total Parameters: {} *****".format(tot_params))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters: {} *****".format(trainable_params))

    print("***** Trainable Parameters Ratio: {} % *****".format(trainable_params/tot_params * 100))

    return model


def prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device='cuda')
        return data.to(**kwargs)
    return data


def loss_func(logits, inputs_ids, model, loss_fct):
    labels = prepare_input_and_label(model, inputs_ids)
    logits = logits[..., :-1, :].contiguous()
    batch_size, seq_len, vocab_size = logits.shape
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(batch_size, -1).sum(dim=-1) # TODO support more objectives
    loss = loss.mean()
    return loss


def prepare_input_and_label(model, inputs_ids):
    # shift right
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    padded_input_tokens = model._extend_labels(inputs_ids)
    labels = padded_input_tokens[..., 1:].contiguous()
    input_tokens = padded_input_tokens[..., :-1].contiguous()
    labels[input_tokens<0] = -100
    return labels


# @torch.no_grad()
# def evaluate(prompt_model, valenc, loss_fct):
#     prompt_model.eval()
#     nlls = []
#     seqlen = args.seqlen
#     n_samples = valenc.input_ids.size(1) // seqlen
#     for i in range(n_samples):
#         inputs_ids = valenc.input_ids[:,i * seqlen:(i+1) * seqlen].cuda()
#         labels = prepare_input_and_label(prompt_model, inputs_ids)
#         try:
#             output = prompt_model(inputs_ids)
#         except:
#             import ipdb; ipdb.set_trace()
#         shift_logits = output.logits[:, :-1, :]
#         loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
#         neg_log_likelihood = loss.float().mean() * seqlen
#         nlls.append(neg_log_likelihood)
#     ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
#     print(f"Perplexity: {ppl.item():3f}")
#     import ipdb; ipdb.set_trace()
#     return ppl.item()


@torch.no_grad()
def evaluate(prompt_model, val_loader, loss_fct, is_llama=True):
    prompt_model.eval()
    nlls = []
    total_samples = 0
    for idx, inputs_ids in tqdm(enumerate(val_loader)):
        if torch.cuda.device_count() == 1:
            inputs_ids = inputs_ids.cuda()
        bs ,seqlen = inputs_ids.shape
        total_samples += bs
        labels = prepare_input_and_label(prompt_model, inputs_ids)
        try:
            output = prompt_model(inputs_ids)
        except:
            import ipdb; ipdb.set_trace()
        shift_logits = output.logits[:, :-1, :]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.shape[-1]), labels.view(-1))
        neg_log_likelihood = loss.float().reshape(bs, -1).mean(dim=-1) * seqlen
        if torch.cuda.device_count() > 1:
            nll = accelerator.gather(neg_log_likelihood.view(1, -1))
        else:
            nll = neg_log_likelihood.view(1, -1)
        nlls.append(nll)
    nlls = torch.hstack(nlls).view(-1)
    ppl = torch.exp(nlls.sum() / (nlls.numel() * seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    return ppl.item()

if __name__ == "__main__":
    args = parser.parse_args()
    ROOT = args.root
    content_write = "="*20+"\n"
    content_write += f"dataset {args.dataset}\t"
    content_write += f"model {args.model}\t"
    content_write += f"seed {args.seed}\t"
    content_write += f"init_from_vocab {args.init_from_vocab}\t"
    content_write += f"eval_every_steps {args.eval_every_steps}\t"
    content_write += f"prompt_lr {args.prompt_lr}\t"
    content_write += f"optimizer {args.optimizer}\t"
    content_write += f"seqlen {args.seqlen}\t"
    content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
    content_write += f"soft_token_num {args.soft_token_num}\t"
    content_write += "\n"

    print(content_write)
    set_seed(args.seed)
    try:
        if not os.path.exists(f"{ROOT}/{args.output_dir}"):
            os.makedirs(f"{ROOT}/{args.output_dir}")
    except FileExistsError:
        pass
    # load model
    if args.model.startswith('facebook/opt'):
        prompt_model = OPTPromptTuningLM.from_pretrained(args.model_name_or_path,
                                                          soft_prompt_path=None,
                                                          n_tokens=args.soft_token_num,
                                                          initialize_from_vocab=args.init_from_vocab,
                                                          torch_dtype=torch.bfloat16)
        prompt_model = freeze_model(prompt_model)
        print(prompt_model.soft_prompt)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        IS_LLAMA = False
    elif args.model.startswith('decapoda-research'):
        prompt_model = LLamaPromptTuningLM.from_pretrained(args.model_name_or_path,
                                                          soft_prompt_path=None,
                                                          n_tokens=args.soft_token_num,
                                                          initialize_from_vocab=args.init_from_vocab,
                                                          torch_dtype=torch.bfloat16)
        prompt_model = freeze_model(prompt_model)
        print(prompt_model.soft_prompt)
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)
        IS_LLAMA = True
    elif args.model.startswith('bigscience/bloom-7b1'):
        prompt_model = BloomPromptTuningLM.from_pretrained(args.model_name_or_path,
                                                          soft_prompt_path=None,
                                                          n_tokens=args.soft_token_num,
                                                          initialize_from_vocab=args.init_from_vocab,
                                                          torch_dtype=torch.bfloat16)
        prompt_model = freeze_model(prompt_model)
        print(prompt_model.soft_prompt)
        IS_LLAMA = False
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    else:
        raise NotImplementedError("currently only support OPT")

    # load dataset
    from torch.utils.data import DataLoader
    if args.dataset == "wikitext2":
        raw_tra_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        raw_val_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        # raw_tst_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='text')
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='text')

        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    elif args.dataset == 'ptb':
        raw_tra_data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        raw_val_data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        raw_tst_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')

        # trainenc = tokenizer(" ".join(raw_tra_data['sentence']), return_tensors='pt')
        # truncate the train sequence
        # tot_num_train_seq = trainenc['input_ids'].size(1) // (args.seqlen*args.per_device_train_batch_size)
        # trainenc['input_ids'] = trainenc['input_ids'][..., :tot_num_train_seq*args.seqlen*args.per_device_train_batch_size]

        # truncate the val sequence
        # valenc = tokenizer("\n\n".join(raw_val_data['sentence']), return_tensors='pt')
        # tot_num_val_seq = valenc['input_ids'].size(1) // args.seqlen
        # valenc['input_ids'] = valenc['input_ids'][..., :tot_num_val_seq*args.seqlen]


        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='sentence')
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='sentence')

        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    elif args.dataset == 'c4':
        raw_tra_data = load_dataset('allenai/c4', 'allenai--c4', 
                                    data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
                                    split='train')
        raw_val_data = load_dataset('allenai/c4', 'allenai--c4', 
                                    data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                                    split='validation')

        # trainenc = tokenizer(" ".join(raw_tra_data['text']), return_tensors='pt')
        # # truncate the train sequence
        # tot_num_train_seq = trainenc['input_ids'].size(1) // (args.seqlen*args.per_device_train_batch_size)
        # trainenc['input_ids'] = trainenc['input_ids'][..., :tot_num_train_seq*args.seqlen*args.per_device_train_batch_size]

        # # truncate the val sequence
        # valenc = tokenizer(' '.join(raw_val_data[:1100]['text']), return_tensors='pt')
        # tot_num_val_seq = valenc['input_ids'].size(1) // args.seqlen
        # valenc['input_ids'] = valenc['input_ids'][..., :tot_num_val_seq*args.seqlen]
        # print('dataset prepare finished')

        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='text', cutoff=5000)
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='text', cutoff=1100)
        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    else:
        raise NotImplementedError("currently only support wikitext2 and rtb dataset")
    
    # build optimizer and lr scheduler
    from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer
    from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
    tot_step = args.max_steps
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": 0.0, # following Openprompt package, we do not use weight decay for soft prompt
        }
    ]

    if args.optimizer.lower() == "adafactor":
        # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        # weight_decay is 1e-5, following the setting of https://arxiv.org/pdf/2104.08691.pdf
        # we set scale_parameter off.
        optimizer = Adafactor(optimizer_grouped_parameters,
                                lr=args.prompt_lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False,
                                weight_decay=1e-5)
          
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.prompt_lr, weight_decay=1e-5) # usually lr = 0.5
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_step_prompt, 
                                                        num_training_steps=tot_step) # usually num_warmup_steps is 500
    else:
        raise NotImplementedError("currently only support AdamW and Adafactor")
            
    tot_loss = 0
    log_loss = 0
    best_val_ppl = float('inf')
    glb_step = 0
    actual_step = 0
    leave_training = False

    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10
    prompt_model.train()

    pbar = tqdm(total=tot_step, desc="Train")
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    if torch.cuda.device_count() > 1:
        accelerator = Accelerator()
        prompt_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            prompt_model, optimizer, train_dataloader, scheduler)
        device = accelerator.device
        val_dataloader = accelerator.prepare(val_dataloader)
    else:
        device = "cuda"
    prompt_model = prompt_model.to(device)
    # prompt_model = BetterTransformer.transform(model)

    # val_ppl = evaluate(prompt_model, val_dataloader, loss_fct)
    # print(f'before training, eval ppl: {val_ppl}')

    for epoch in range(1000000): # 1000000
        print(f"Begin epoch {epoch}")
        prompt_model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.device_count() > 1:
                input_ids = batch
            else:
                input_ids = batch.cuda()
            output = prompt_model(input_ids)
            logits = output.logits
            loss = loss_func(logits, input_ids, prompt_model, loss_fct)
            if torch.cuda.device_count() > 1:
                accelerator.backward(loss)
            else:
                loss.backward()
            tot_loss += loss.item()
            actual_step += 1
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1

            # set progress bar
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss) / pbar_update_freq
                pbar.update(10)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss
            # update
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


            tot_train_time += time.time()

            if glb_step >0 and glb_step % args.eval_every_steps == 0:
                val_ppl = evaluate(prompt_model, val_dataloader, loss_fct, IS_LLAMA)
                print(f'{val_ppl}: val_ppl')
                if val_ppl <= best_val_ppl:
                    if isinstance(prompt_model, torch.nn.parallel.DistributedDataParallel):
                        unwrapped_model = accelerator.unwrap_model(prompt_model)
                        accelerator.save({
                            "model": unwrapped_model.soft_prompt.state_dict(),
                            "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
                        }, f"{ROOT}/{args.output_dir}/best.pth")
                    else:
                        import ipdb; ipdb.set_trace()
                        torch.save({"model": prompt_model.soft_prompt.state_dict(), 
                                    "optimizer": optimizer.state_dict(),
                                    },f"{ROOT}/{args.output_dir}/best.pth")
                    best_val_ppl = val_ppl
                    print(f"best val acc: {best_val_ppl}")

                acc_traces.append(val_ppl)
                print("Glb_step {}, val_ppl {}, average time {}".format(glb_step, val_ppl, tot_train_time/actual_step ), flush=True)
                prompt_model.train()
            if glb_step > args.max_steps:
                leave_training = True
                break

        if leave_training:
            break

    # n_batchs = trainenc['input_ids'].size(1) // (args.seqlen * args.per_device_train_batch_size)
    # for epoch in range(1000000): # 1000000
    #     print(f"Begin epoch {epoch}")
    #     prompt_model.train()
    #     for i in range(n_batchs):
    #         input_ids = trainenc.input_ids[:, i*args.seqlen*args.per_device_train_batch_size:(i+1)*args.seqlen*args.per_device_train_batch_size].cuda()
    #         input_ids = input_ids.reshape(args.per_device_train_batch_size, args.seqlen)
    #         # shuffle the input
    #         input_ids = input_ids[torch.randperm(input_ids.shape[0])]
    #         output = prompt_model(input_ids)
    #         logits = output.logits
    #         loss = loss_func(logits, input_ids, prompt_model, loss_fct)
    #         loss.backward()
    #         tot_loss += loss.item()
    #         actual_step += 1
            
    #         # clip gradient
    #         torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
    #         glb_step += 1

    #         # set progress bar
    #         if glb_step % pbar_update_freq == 0:
    #             aveloss = (tot_loss - log_loss) / pbar_update_freq
    #             pbar.update(10)
    #             pbar.set_postfix({'loss': aveloss})
    #             log_loss = tot_loss
            
    #         # update
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         scheduler.step()


    #         tot_train_time += time.time()

    #         if glb_step >0 and glb_step % args.eval_every_steps == 0:
    #             # TODO eval 
    #             val_ppl = evaluate(prompt_model, valenc, loss_fct)
    #             print(f'{val_ppl}: val_ppl')
    #             if val_ppl <= best_val_ppl:
    #                 torch.save(prompt_model.state_dict(),f"{ROOT}/{args.output_dir}/best.ckpt")
    #                 best_val_ppl = val_ppl
    #                 print(f"best val acc: {best_val_ppl}")

    #             acc_traces.append(val_ppl)
    #             print("Glb_step {}, val_ppl {}, average time {}".format(glb_step, val_ppl, tot_train_time/actual_step ), flush=True)
    #             prompt_model.train()
    #         if glb_step > args.max_steps:
    #             leave_training = True
    #             break

    #     if leave_training:
    #         break