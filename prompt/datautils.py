import numpy as np
import torch
from prompt import llama_loader
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, args, col_key, mode="train", cutoff=None):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.col_key = col_key
        self.cutoff = cutoff

        if self.mode == "train":
            self.encodings = self.process_data(data)
        else:
            self.encodings = self.process_data(data, is_val=True)

    def process_data(self, data, is_val=False):
        if is_val:
            if self.cutoff is None:
                enc = self.tokenizer(" ".join(data[self.col_key]), return_tensors='pt')
            else:
                enc = self.tokenizer(" ".join(data[:self.cutoff][self.col_key]), return_tensors='pt')
            tot_num_seq = enc['input_ids'].size(1) // self.args.seqlen
            enc['input_ids'] = enc['input_ids'][..., :tot_num_seq*self.args.seqlen]
        else:
            if self.cutoff is None:
                enc = self.tokenizer(" ".join(data[self.col_key]), return_tensors='pt')
            else:
                enc = self.tokenizer(" ".join(data[:self.cutoff][self.col_key]), return_tensors='pt')
            tot_num_seq = enc['input_ids'].size(1) // (self.args.seqlen*self.args.per_device_train_batch_size)
            enc['input_ids'] = enc['input_ids'][..., :tot_num_seq*self.args.seqlen*self.args.per_device_train_batch_size]
        
        return enc

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids'][0, idx*self.args.seqlen:(idx+1)*self.args.seqlen]
        return input_ids

    def __len__(self):
        return self.encodings['input_ids'].size(1) // self.args.seqlen
    


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model, llama=False, return_tokenizer=False):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    if llama:
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(model, use_fast=False)
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    if return_tokenizer:
        return trainloader, testenc, tokenizer
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, llama=False, return_tokenizer=False):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    if llama:
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(model, use_fast=False)
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    if return_tokenizer:
        return trainloader, testenc, tokenizer
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model, llama=False, return_tokenizer=False):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    if llama:
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(model, use_fast=False)
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)
    if return_tokenizer:
        return trainloader, valenc, tokenizer
    return trainloader, valenc 


def get_ptb_new(nsamples, seed, seqlen, model, llama=False, return_tokenizer=False):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    if llama:
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(model, use_fast=False)
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    if return_tokenizer:
        return trainloader, testenc, tokenizer
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model, llama=False, return_tokenizer=False):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    if llama:
        tokenizer = llama_loader.LLaMATokenizer.from_pretrained(model, use_fast=False)
    else:
        from transformers import AutoTokenizer 
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)
    if return_tokenizer:
        return trainloader, valenc, tokenizer
    return trainloader, valenc


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', llama=False, return_tokenizer=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, llama, return_tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, llama, return_tokenizer)
        return get_ptb(nsamples, seed, seqlen, model, llama, return_tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, llama, return_tokenizer)
        return get_c4(nsamples, seed, seqlen, model, llama, return_tokenizer)
