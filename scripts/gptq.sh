dataset=c4
model=decapoda-research/llama-7b-hf
soft_token_num=100

for opt in adamw; do
for lr in 0.001; do
for steps in 30000; do
CUDA_VISIBLE_DEVICES=$1 python soft_prompt_learning.py \
    --model_name_or_path $2 \
    --model ${model} \
    --dataset ${dataset} \
    --eval_every_steps 100 \
    --seqlen 1024 \
    --soft_token_num ${soft_token_num} \
    --prompt_lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --output_dir ./gptq/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${dataset} \
    --per_device_train_batch_size 2 2>&1 | tee ./logs/log_gptq_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt
done 
done
done

# | tee ./logs/log_${opt}_lr${lr}_${dataset}_steps${steps}.txt
# --per_device_eval_batch_size 1 \