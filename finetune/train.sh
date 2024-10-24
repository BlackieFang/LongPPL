#!/bin/bash

# run `accelerate config` first. pass --deepspeed to finetune.py if using DeepSpeed

accelerate launch finetune.py \
    --output-dir output/llama2-pg19-PI-longce \
    --model Llama-2-7b-hf \
    --deepspeed \
    --max-train-steps 200 \
    --scaling-factor 8.0 \
    --dataset datasets/pg19 \
    --save-steps 50 \
    --gradient-accumulate-every 8 \

# accelerate launch finetune.py \
#     --output-dir output/llama2-pg19-eabf-longce \
#     --model Llama-2-7b-hf \
#     --deepspeed \
#     --max-train-steps 200 \
#     --use-eabf \
#     --dataset datasets/pg19 \
#     --save-steps 50 \
#     --gradient-accumulate-every 8 \

# accelerate launch finetune.py \
#     --output-dir output/llama2-pg19-arxiv-longce \
#     --model Llama-2-7b-hf \
#     --deepspeed \
#     --max-train-steps 200 \
#     --use-eabf \
#     --dataset datasets/pile_arxiv \
#     --save-steps 50 \
#     --gradient-accumulate-every 1 \
