import llama_loader
import ipdb
tokenizer = llama_loader.LLaMATokenizer.from_pretrained('decapoda-research/llama-7b-hf')
model = llama_loader.LLaMAForCausalLM.from_pretrained('decapoda-research/llama-7b-hf')
print(model)
ipdb.set_trace()
# print(tokenizer.decode(model.generate(tokenizer('Yo mama', return_tensors = "pt")["input_ids"])[0]))