import argparse
import copy
import torch
import os
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed, DummyOptim, DummyScheduler
from tqdm import tqdm
from transformers import set_seed, default_data_collator, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM


def loss_weight(model, input_ids, trunc_len=4096, internal=1024, thre=5):
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    _, max_len = input_ids.shape

    output_full = model(input_ids)
    loss = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:])

    loss_discrepancy = torch.ones(input_ids.shape[-1], device=input_ids.device)

    with torch.no_grad():
        for i, start_token in enumerate(range(0, max_len-trunc_len, internal)):
            if start_token+trunc_len+internal > max_len:
                internal = max_len-start_token-trunc_len

            input_ids_short = input_ids[:, start_token: start_token+trunc_len+internal]
            output_short = model(input_ids_short)

            loss_full = loss[start_token+trunc_len-1: start_token+trunc_len+internal-1]
            loss_short = loss_f(output_short.logits[0, trunc_len-1: trunc_len+internal-1, :], input_ids_short[0, trunc_len: trunc_len+internal])
            loss_discrepancy[start_token+trunc_len: start_token+trunc_len+internal] = torch.exp(loss_short - loss_full).squeeze()

    weight = torch.clamp(loss_discrepancy[1:], max=thre)
        
    return weight, loss

def load_model(args):
    if args.use_eabf:
        if "Llama" in args.model:
            import patch.EABF as eabf
            model = LlamaForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                rope_scaling={"type":"eabf", "factor": 4.0}
            )
            eabf.apply_eabf(model)
        elif "Mistral" in args.model:
            import patch.EABF_mistral as eabf_mistral
            model = MistralForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                rope_scaling={"type":"eabf", "factor": 4.0},
                sliding_window=None
            )
            eabf_mistral.apply_eabf(model)
        else:
            raise NotImplementedError
    else:
        config = AutoConfig.from_pretrained(args.model)
        config.rope_scaling = {
            "type": "linear",
            "factor": args.scaling_factor,
            "original_max_position_embeddings": args.original_max_position_embeddings
        }
        config.rope_theta = args.rope_theta
        config.max_position_embeddings = int(args.scaling_factor * args.original_max_position_embeddings) \
            if not args.max_position_embeddings else args.max_position_embeddings

        if "Llama" in args.model:
            model = LlamaForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                config=config,
            )
        elif "Mistral" in args.model:
            model = MistralForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                config=config,
                sliding_window=None,
            )
        else:
            raise NotImplementedError
    
    return model


def main(args):
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout]
    )
    accelerator.init_trackers(
        project_name=args.wandb if args.wandb else "LongCE",
    )
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    model = load_model(args)
    print(model.config)

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        try:
            train_dataset = train_dataset["train"]
        except:
            train_dataset = train_dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size
    )

    if args.deepspeed:
        optim = DummyOptim(model.parameters(), lr=args.learning_rate)
        scheduler = DummyScheduler(
            optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
        model, optim, train_loader, scheduler = accelerator.prepare(
            model, optim, train_loader, scheduler
        )
    else:
        model = accelerator.prepare(model)
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if args.lr_schedule == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim, num_training_steps=args.max_train_steps, num_warmup_steps=args.warmup_steps)
        elif args.lr_schedule == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optim, num_warmup_steps=args.warmup_steps)
        optim, train_loader, scheduler = accelerator.prepare(
            optim, train_loader, scheduler)

    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)
    total_batch_size = (
        args.batch_size * accelerator.num_processes * args.gradient_accumulate_every
    )

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    accelerator.print(f"Total batch size: {total_batch_size}")
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = (
            int(training_difference.replace("step_", ""))
        )

    if args.resume_from_checkpoint and resume_step is not None:
        train_loader = accelerator.skip_first_batches(
            train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)
        accelerator.print(f"Resuming training from step {resume_step}")

    loss_file = open(args.log_loss, "a" if args.resume_from_checkpoint else "w") if args.log_loss and accelerator.is_main_process else None
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')

    # if not args.save_only:
    model.train()
    for step, batch in enumerate(train_loader):
        stride = 20000 if 'arxiv' in args.dataset else 32768
        text = batch['text']
        tokenizer.pad_token = tokenizer.eos_token
        input_enc = tokenizer(text, padding="max_length", max_length=stride,return_tensors="pt")
        input_seq = input_enc['input_ids'].to(model.device)
        seq_len = input_seq.shape[-1]
            
        for i in range(0, seq_len, stride):
            if i+stride > seq_len:
                break
            input_ids = input_seq[:, i: i+stride]

            loss_log = None
            with accelerator.accumulate(model):
                if args.loss_type == 'ce':
                    loss = model(input_ids, labels=input_ids).loss
                elif args.loss_type == 'longce':
                    weight, loss_origin = loss_weight(model, input_ids, trunc_len=4096, internal=1024, thre=args.threshold)
                    loss = torch.mean(loss_origin * weight)
                else:
                    raise NotImplementedError
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    loss_log = {"loss": loss.item()}
                    accelerator.log(loss_log, step=completed_steps)
                    if loss_file is not None:
                        loss_file.write(f"{loss_log['loss']},")
                        loss_file.flush()
                    if isinstance(args.grad_norm, float):
                        accelerator.clip_grad_norm_(
                            model.parameters(), args.grad_norm)

                optim.step()
                scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                if loss_log is not None:
                    progress_bar.set_postfix(loss_log)
                completed_steps += 1

                if isinstance(args.checkpointing_steps, int) and completed_steps > 0:
                    if completed_steps % args.checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(
                                args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

            if completed_steps % args.save_steps == 0 and completed_steps > 0:
                output_dir = os.path.join(args.output_dir, f"{completed_steps}ep")
                accelerator.print(f"Saving model to {output_dir}")

                accelerator.wait_for_everyone()

                if args.deepspeed:
                    state_dict = accelerator.get_state_dict(model)
                else:
                    full_state_dict_config = FullStateDictConfig(
                        offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                        state_dict = accelerator.get_state_dict(model, unwrap=False)

                accelerator.unwrap_model(model).save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=state_dict,
                )

                accelerator.print(f"Saving Finished")

                if accelerator.is_main_process:
                    os.remove(os.path.join(output_dir, "model.safetensors"))

            if completed_steps >= args.max_train_steps:
                break

        if completed_steps >= args.max_train_steps:
            break
        

    accelerator.print(f"Training Finished")
    accelerator.end_training()



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--resume-from-checkpoint", type=str)
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--save-steps", type=int, default=50)
    args.add_argument("--warmup-steps", type=int, default=20)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--grad-norm", action="store_true")
    args.add_argument("--model", type=str, default="Llama-2-7b-hf")
    args.add_argument("--scaling-factor", type=float, default=8.0)
    args.add_argument("--rope-theta", type=float, default=10000.0)
    args.add_argument("--dataset", type=str, default="pg19")
    args.add_argument("--deepspeed", action="store_true")
    args.add_argument("--max-position-embeddings", type=int)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant"], default="linear")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--original-max-position-embeddings", type=int, default=4096)
    args.add_argument("--loss-type", type=str, choices=['ce', 'longce'], default="longce")
    args.add_argument("--threshold", type=float, default=5.0)
    args.add_argument("--trunc-len", type=int, default=4096)
    args.add_argument("--internal", type=int, default=1024)
    args.add_argument("--use-eabf", action="store_true")
    main(args.parse_args())
