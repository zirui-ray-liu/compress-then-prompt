
import llama_loader

MODEL = 'decapoda-research/llama-7b-hf'
# MODEL = 'decapoda-research/llama-13b-hf'
# MODEL = 'decapoda-research/llama-30b-hf'
# MODEL = 'decapoda-research/llama-65b-hf'

tokenizer = llama_loader.LLaMATokenizer.from_pretrained(MODEL)
model = llama_loader.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage = True)
model.to('cuda')

batch = tokenizer("Yo mama", return_tensors = "pt")
print(tokenizer.decode(model.generate(batch["input_ids"].cuda(), max_length=100)[0]))

# Expected output: "Yo mama is so fat, she has to buy two seats on the plane"