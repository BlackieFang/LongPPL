import argparse
import datasets
import gc
import sys
import torch
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

def merge_intervals(intervals):
    if intervals.size(0) == 0:
        return intervals

    start = intervals[:, 0]
    end = intervals[:, 1]
    adjacent = (start[1:] - end[:-1]) == 0

    keep_start_mask = torch.cat([torch.tensor([True]), ~adjacent])
    merged_start = start[keep_start_mask]
    keep_end_mask = torch.cat([~adjacent, torch.tensor([True])])
    merged_end = end[keep_end_mask]

    merged_intervals = torch.stack([merged_start, merged_end], dim=1)
    
    return merged_intervals 

def find_key_token(model, origin_input_ids, trunc_len, internal, tokenizer, input_text, save_path):
    text_encoded = tokenizer(input_text, return_tensors="pt", return_offsets_mapping=True)
    input_ids = text_encoded['input_ids'].to(origin_input_ids.device)
    
    with torch.no_grad():
        output_full = model(input_ids)
    
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    _, max_len = input_ids.shape
    key_tokens = []

    with torch.no_grad():
        for i, start_token in enumerate(range(0, max_len-trunc_len, internal)):
            if start_token+trunc_len+internal > max_len:
                internal = max_len-start_token-trunc_len

            input_ids_short = input_ids[:, start_token: start_token+trunc_len+internal]
            output_short = model(input_ids_short)

            loss_full = loss_f(output_full.logits[0, start_token+trunc_len-1: start_token+trunc_len+internal-1, :], input_ids[0, start_token+trunc_len: start_token+trunc_len+internal])
            loss_short = loss_f(output_short.logits[0, trunc_len-1: trunc_len+internal-1, :], input_ids_short[0, trunc_len: trunc_len+internal])

            loss_discrepancy = (torch.logical_and((loss_short - loss_full) > 2, loss_full < 2)).squeeze()

            for i, is_key in enumerate(loss_discrepancy):
                if is_key:
                    key_tokens.append(start_token+trunc_len+i)
                
    key_text_intervals = merge_intervals(text_encoded['offset_mapping'][0, key_tokens])

    with open(save_path, "w", encoding="utf-8") as f:
        slices_str = ";".join([f"[{element[0]}, {element[1]}]" for element in key_text_intervals])
        f.write(slices_str)

    return key_text_intervals

def load_key_token(save_path, input_text):
    with open(save_path, "r+", encoding="utf-8") as f:
        for line in f.readlines():
            key_slices_str = line.split(';')
            key_text_slices = []
            for key_slice in key_slices_str:
                key_text_slices.append(eval(key_slice))
            return key_text_slices

def cal_overlap(offset_mapping, key_text_slices, text=None):
    if key_text_slices is None:
        return None

    key_tokens = []
    i, j = 0, 0
    
    while i < len(offset_mapping) and j < len(key_text_slices):
        a_start, a_end = offset_mapping[i]
        b_start, b_end = key_text_slices[j]

        if a_start >= b_start and a_end <= b_end:
            key_tokens.append(i-1)
            i += 1
        elif a_start < b_start:
            i += 1
        else:
            j += 1

    return key_tokens



def perplexity(model, evaluator_model, input_ids, tokenizer, evaluator_tokenizer, input_text, save_path, offset_mapping, trunc_len=4096, internal=1024):
    if evaluator_model is not None:
        key_text_slices = find_key_token(evaluator_model, input_ids, trunc_len, internal, evaluator_tokenizer, input_text, save_path)
    else:
        key_text_slices = load_key_token(save_path, input_text)

    key_tokens = cal_overlap(offset_mapping, key_text_slices, text=input_text)
    
    with torch.no_grad():
        output_full = model(input_ids)

    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
    
    if key_tokens is None or len(key_tokens) == 0:
        return None, np.exp(loss_overall.mean()), None, input_ids.shape[-1]

    loss_key = loss_overall[key_tokens]

    return np.exp(loss_key.mean()), np.exp(loss_overall.mean()), len(key_tokens), input_ids.shape[-1]

