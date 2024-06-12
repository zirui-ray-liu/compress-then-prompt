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
from prompt import LLamaPromptTuningLM, OPTPromptTuningLM, llama_loader, TextDataset
from transformers import AutoTokenizer, set_seed, default_data_collator, AutoModelForCausalLM

from datasets import load_dataset
from typing import Any, Union
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType


parser = argparse.ArgumentParser("")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--dataset",type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", default=20000, type=int)
parser.add_argument("--lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=500)
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--eval_every_steps", type=int, default=500)
parser.add_argument("--optimizer", type=str, default="Adafactor")
parser.add_argument("--dataloader_num_workers", type=int, default=16)
parser.add_argument("--dataloader_pin_memory", action="store_true")
parser.add_argument("--seqlen", type=int, default=1024)
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_r", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.1)



def freeze_model(model):
    for n, m in model.named_parameters():
        if "lora" in n.lower():
            m.requires_grad = True 
        else:
            m.requires_grad = False

    tot_params = sum(p.numel() for p in model.parameters())
    print("***** Model Total Parameters: {} *****".format(tot_params))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("***** Model Trainable Parameters: {} *****".format(trainable_params))

    print("***** Trainable Parameters Ratio: {} % *****".format(trainable_params/tot_params * 100))

    return model


def loss_func(logits, inputs_ids, model, loss_fct):
    labels = inputs_ids[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    batch_size, seq_len, vocab_size = logits.shape
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(batch_size, -1).sum(dim=-1) # TODO support more objectives
    loss = loss.mean()
    return loss


def extract_lora_layers(model):
    lora_layers = {}
    for name, param in model.named_parameters():
        if 'lora' in name:  # Assuming 'lora' is in the names of LoRA layers
            lora_layers[name] = param.data
    return lora_layers


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
        with torch.amp.autocast(dtype=torch.float16, device_type="cuda"): 
            output = prompt_model(inputs_ids)
        #  prompt_model.base_model.model.transformer.h[0].self_attention.query_key_value
        labels = inputs_ids[..., 1:].contiguous()
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
    content_write += f"eval_every_steps {args.eval_every_steps}\t"
    content_write += f"lr {args.lr}\t"
    content_write += f"optimizer {args.optimizer}\t"
    content_write += f"seqlen {args.seqlen}\t"
    content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
    content_write += f"lora_r {args.lora_r}\t"
    content_write += f"lora_alpha {args.lora_alpha}\t"
    content_write += f"lora_dropout {args.lora_dropout}\t"
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
        raise NotImplementedError
        IS_LLAMA = False
    elif args.model.startswith('decapoda-research'):
        raise NotImplementedError
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)
        IS_LLAMA = True
    elif args.model.startswith('bigscience/bloom-7b1'):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                          torch_dtype=torch.float16)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["query", "value"],
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model = freeze_model(model)
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
            "params": [p for n, p in model.named_parameters() if "lora" in n.lower()],
            "weight_decay": 0.0, # following Openprompt package, we do not use weight decay for soft prompt
        }
    ]

    if args.optimizer.lower() == "adafactor":
        # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        # weight_decay is 1e-5, following the setting of https://arxiv.org/pdf/2104.08691.pdf
        # we set scale_parameter off.
        optimizer = Adafactor(optimizer_grouped_parameters,
                                lr=args.lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False,
                                weight_decay=1e-5)
          
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=1e-5) # usually lr = 0.5
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
    model.train()

    pbar = tqdm(total=tot_step, desc="Train")
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    if torch.cuda.device_count() > 1:
        accelerator = Accelerator()
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler)
        device = accelerator.device
        val_dataloader = accelerator.prepare(val_dataloader)
    else:
        device = "cuda"
    model = model.to(device)
    val_ppl = evaluate(model, val_dataloader, loss_fct)
    print(f'before training, eval ppl: {val_ppl}')

    for epoch in range(1000000): # 1000000
        print(f"Begin epoch {epoch}")
        model.train()
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.device_count() > 1:
                input_ids = batch
            else:
                input_ids = batch.cuda()
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"): 
                output = model(input_ids)
            logits = output.logits
            loss = loss_func(logits, input_ids, model, loss_fct)
            if torch.cuda.device_count() > 1:
                accelerator.backward(loss)
            else:
                loss.backward()
            tot_loss += loss.item()
            actual_step += 1
            
            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                val_ppl = evaluate(model, val_dataloader, loss_fct, IS_LLAMA)
                print(f'{val_ppl}: val_ppl')
                if val_ppl <= best_val_ppl:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        unwrapped_model = accelerator.unwrap_model(model)
                        lora_states = extract_lora_layers(unwrapped_model)
                        accelerator.save({
                            "model": lora_states,
                            "optimizer": optimizer.optimizer.state_dict() # optimizer is an AcceleratedOptimizer object
                        }, f"{ROOT}/{args.output_dir}/best.pth")
                    else:
                        lora_states = extract_lora_layers(model)
                        torch.save({"model": lora_states, 
                                    "optimizer": optimizer.state_dict(),
                                    },f"{ROOT}/{args.output_dir}/best.pth")
                    best_val_ppl = val_ppl
                    print(f"best val acc: {best_val_ppl}")

                acc_traces.append(val_ppl)
                print("Glb_step {}, val_ppl {}, average time {}".format(glb_step, val_ppl, tot_train_time/actual_step ), flush=True)
                model.train()
            if glb_step > args.max_steps:
                leave_training = True
                break

        if leave_training:
            break