import argparse
import os
import ipdb
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, OPTForCausalLM
from datasets import load_dataset 
import torch
import torch.nn as nn
from prompt import LLamaPromptTuningLM, llama_loader, OPTPromptTuningLM
from prompt.modelutils import get_llama


parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', type=str, default= "decapoda-research/llama-7b-hf")
parser.add_argument('--model-name-or-path', type=str, required=True)
# parser.add_argument('--model-name-or-path', type=str, default= "decapoda-research/llama-7b-hf")
parser.add_argument('--dtype', type = str, default = "auto")
args = parser.parse_args()


@torch.no_grad()
def evaluate(prompt_model, seqlen, tokenizer):
    prompt_model.eval()
    hard_prompt = 'The weight matrix inside the language model contains errors. Make any necessary adjustments to ensure optimal performance.\n'
    # hard_prompt = 'The weight matrix inside the language model has been quantized. Make any necessary adjustments to ensure optimal performance.'
    # hard_prompt = ''
    n_tokens = len(hard_prompt)
    question = 'Please answer the following question: Who won the Nobel Peace Prize in 2009?'
    input = hard_prompt + question 
    # input = question
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    generate_ids = model.generate(input_ids, max_length=seqlen)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)


if args.dtype == 'auto':
    dtype = 'auto'
elif args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    raise NotImplementedError

model = get_llama(args.model_name_or_path)
tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)

print(model.dtype)

model.seqlen = model.config.max_position_embeddings
model.seqlen = 1024
model.cuda()
model.eval()


print(f'model_name_or_path: {args.model_name_or_path}\n')

evaluate(model, model.seqlen, tokenizer)