import argparse
import os
import ipdb
from transformers import AutoTokenizer, LlamaForCausalLM, OPTForCausalLM
from datasets import load_dataset 
import torch
import torch.nn as nn
from prompt import LLamaPromptTuningLM, llama_loader, OPTPromptTuningLM
from prompt.modelutils import get_llama


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', type=str, default= "decapoda-research/llama-7b-hf")
parser.add_argument('--model-name-or-path', type=str, required=True)
parser.add_argument('--ckpt', type=str, default=None)
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/baseline/ptb/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/adamw_lr0.001_steps30000/c4/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/unpruned/ptb/best.ckpt")
parser.add_argument('--dataset', type = str, default = "wikitext2")
parser.add_argument('--dtype', type = str, default = "auto")
parser.add_argument('--ntoken', type = int, default = 50)
args = parser.parse_args()


def prepare_input_and_label(model, inputs_ids):
    # shift right
    if hasattr(model, 'n_tokens'):
        padded_input_tokens = model._extend_labels(inputs_ids)
    else:
        padded_input_tokens = inputs_ids
    labels = padded_input_tokens[..., 1:].contiguous()
    input_tokens = padded_input_tokens[..., :-1].contiguous()
    labels[input_tokens<0] = -100
    return labels


@torch.no_grad()
def evaluate(prompt_model, valenc, loss_fct, seqlen):
    prompt_model.eval()
    nlls = []
    if not isinstance(valenc, torch.Tensor):
        valenc = valenc.input_ids
    n_samples = valenc.size(1) // seqlen
    for i in range(n_samples):
        inputs_ids = valenc[:,i*seqlen:(i+1)*seqlen].cuda()
        labels = prepare_input_and_label(prompt_model, inputs_ids)
        try:
            # if isinstance(prompt_model, LLamaPromptTuningLM):
            #     output = prompt_model.forward_with_soft_prompt(inputs_ids)
            # else:
            #     output = prompt_model(inputs_ids)
            output = prompt_model(inputs_ids)
        except:
            import ipdb; ipdb.set_trace()
        shift_logits = output.logits[:, :-1, :]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        neg_log_likelihood = loss.float().mean() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
    return ppl.item()


if args.dtype == 'auto':
    dtype = 'auto'
elif args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    raise NotImplementedError

if 'llama' in args.model:
    if args.ckpt is None:
        model = get_llama(args.model_name_or_path)
    else:
        model = LLamaPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=args.ntoken)
    tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)


elif 'opt' in args.model:
    if args.ckpt is None:
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = OPTPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=args.ntoken)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)


print(model.dtype)

if args.ckpt is not None:
    state_dicts = torch.load(args.ckpt)
    soft_prompt_state_dict = state_dicts['model']
    model.soft_prompt.load_state_dict(soft_prompt_state_dict)
model.seqlen = model.config.max_position_embeddings
model.seqlen = 1024
model.cuda()
model.eval()

if args.dataset == "wikitext2":
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    valdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    valenc = tokenizer("\n\n".join(valdata['text']), return_tensors='pt')

elif args.dataset == "ptb":
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    valenc = tokenizer(" ".join(valdata['sentence']), return_tensors='pt')

elif args.dataset == "c4":
    # follow the implementation in datautils.py of SparseGPT.
    valdata = load_dataset('allenai/c4', 'allenai--c4', 
                                        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                                        split='validation')
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')



loss_fct = nn.CrossEntropyLoss(reduction='none')

print(f'dataset {args.dataset}\nckpt: {args.ckpt}\nmodel_name_or_path: {args.model_name_or_path}\n')

print(f"eval Perplexity:", evaluate(model, valenc, loss_fct, model.seqlen))

# print(f"eval Perplexity:", opt_eval(model, valenc, 'cuda'))


if args.dataset != 'c4':
    testenc = testenc.input_ids
    print(f"test Perplexity:", evaluate(model, testenc, loss_fct, model.seqlen))
    # print(f"test Perplexity:", opt_eval(model, testenc, 'cuda'))
