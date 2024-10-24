import argparse
import datasets
import gc
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from longppl import *
import os


def compute_perplexity(
    encodings, model, evaluator_model, tokenizer, evaluator_tokenizer, evaluator_name, trunc_len, internal, tokenized, device=None, max_length=None, aggressive_memory=False
):
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoded_texts = [x[0:max_length-1] for x in encodings["input_ids"]]
    attn_masks = [x[0:max_length-1] for x in encodings["attention_mask"]]

    pbar = tqdm(total=len(encoded_texts))
    longppls, ppls, nums_key_token, nums_token = [], [], [], []

    def convert_tokenized_to_text(tokenized_input, llama_path):
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_path)
        text = llama_tokenizer.batch_decode(tokenized_input)
        return text

    for encoding_index in range(0, len(encoded_texts)):
        tokenized_input = torch.tensor(encoded_texts[encoding_index:encoding_index+1]).to(device)
        if tokenized:
            text = convert_tokenized_to_text(tokenized_input, args.llama_path)
            tokenized_input = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        input_ids = tokenized_input['input_ids'].to(device)
        offset_mapping = tokenized_input['offset_mapping'][0]
        save_path = f'key_text/{evaluator_name}/slice_{encoding_index}.txt'

        with torch.no_grad():
            longppl, ppl, n_key_token, n_token = perplexity(
                model=model,
                evaluator_model=evaluator_model,
                input_ids=input_ids, 
                tokenizer=tokenizer, 
                evaluator_tokenizer=evaluator_tokenizer, 
                input_text=text[0], 
                save_path=save_path, 
                offset_mapping=offset_mapping, 
                trunc_len=trunc_len, 
                internal=internal
            )
            
        if aggressive_memory:
            outputs = None
            input_ids = None
            gc.collect()
            torch.cuda.empty_cache()

        if longppl is not None:
            longppls.append(longppl)
            nums_key_token.append(n_key_token)
        ppls.append(ppl)
        nums_token.append(n_token)

        longppl = (np.stack(longppls) * np.stack(nums_key_token)).sum() / np.stack(nums_key_token).sum()
        ppl = (np.stack(ppls) * np.stack(nums_token)).sum() / np.stack(nums_token).sum()

        pbar.set_postfix(longppl=longppl, ppl=ppl)
        pbar.update(1)

    return {"longppl": longppl, "ppl": ppl}



def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    evaluator_name = args.evaluator_name
    evaluator_path = args.evaluator_model
    if args.mode == 'online':
        evaluator_model = AutoModelForCausalLM.from_pretrained(evaluator_path, torch_dtype=torch.bfloat16, device_map="auto")
    elif args.mode == 'offline':
        evaluator_model = None
    evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator_path)

    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            example["offsets_mapping"] = tokenized["offsets_mapping"]
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
    if args.samples:
        input_texts = input_texts['test'][:args.samples]
    

    ppl = compute_perplexity(
        model=model, 
        evaluator_model=evaluator_model, 
        tokenizer=tokenizer, 
        evaluator_tokenizer=evaluator_tokenizer, 
        evaluator_name=evaluator_name,
        encodings=input_texts,
        max_length=args.max_length,
        aggressive_memory=args.aggressive_memory,
        trunc_len=args.trunc_len,
        internal=args.internal,
        tokenized=args.tokenized
    )
    print(f"{args.model}: longppl: {ppl['longppl']}, ppl: {ppl['ppl']}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--evaluator-model", type=str)
    parser.add_argument("--evaluator-name", type=str, help='To use the offline key tokens we provided, set it to Qwen2-72B-Instruct, Mistral-Large-Instruct-2407, or Meta-Llama-3.1-8B', default="Meta-Llama-3.1-8B")
    parser.add_argument("--mode", type=str, choices=['online', 'offline'], default='offline')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--aggressive-memory", action="store_true")
    parser.add_argument("--trunc-len", type=int, default=4096)
    parser.add_argument("--internal", type=int, default=1024)
    parser.add_argument("--llama-path", type=str, default="Llama-2-7b-hf")
    main(parser.parse_args())