from ast import arg
import os
import time
import argparse
import json
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, )
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.trainer_utils import seed_worker

from transformers.optimization import get_cosine_schedule_with_warmup
from dataset import make_supervised_data_module
from accelerate.data_loader import skip_first_batches
#from textaugment import Wordnet
import wandb

from utils import (
    get_fsdp_wrapped_empty_model,
    load_model_opt_scheduler_states_fsdp,
    load_state_dict_fsdp,
    save_model_opt_scheduler_states_fsdp,
    add_padding_token,
)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_empty_model(model_config_path,
                    add_tokens=1,
                    wrapped_class=None,
                    hack=False,
                    train_option=0):
    model_config = transformers.AutoConfig.from_pretrained(model_config_path)
    model_config.vocab_size += add_tokens
    return get_fsdp_wrapped_empty_model(model_config,
                                        wrapped_class,
                                        hack=hack,
                                        train_option=train_option)


def get_model_opt_scheduler(
    added_tokens,
    model_config_path,
    max_steps=1000,
    warmup_ratio=0.03,
    weight_decay=0.0,
    lr=2e-5,
    wrapped_class=None,
    hack=False,
    train_option=0,
):
    model = get_empty_model(
        model_config_path,
        add_tokens=added_tokens,
        wrapped_class=wrapped_class,
        hack=hack,
        train_option=train_option,
    )
    opt = torch.optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(opt,
                                                int(max_steps * warmup_ratio),
                                                num_training_steps=max_steps)
    return model, opt, scheduler


def get_dataloader_and_sampler(train_dataset,
                               data_collator,
                               batch_size,
                               rank,
                               world_size=4):
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        seed=0,
    )
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            sampler=sampler,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
            worker_init_fn=seed_worker,
        ),
        sampler,
    )


def get_class_from_class_name(class_name):
    if class_name == "LlamaDecoderLayer":
        return LlamaDecoderLayer
    elif class_name == "OPTDecoderLayer":
        return OPTDecoderLayer
    else:
        raise ValueError(f"Unknown class name {class_name}")


@record
def fsdp_main(rank, world_size, args):
    setup(rank, world_size, args.port)
    if rank == 0:
        if args.wandb:
            wandb.init(
                project=args.wb_project,
                name=args.wb_name,
                config=args,
                resume=args.resume,
            )

    torch.cuda.set_device(rank)
    wrapped_class = get_class_from_class_name(args.wrapped_class_name)
    model, opt, scheduler = get_model_opt_scheduler(
        added_tokens=args.added_tokens,
        model_config_path=args.model_config_path,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr=args.lr,
        wrapped_class=wrapped_class,
        hack=args.hack,
        train_option=args.train_option,
    )
    if args.resume:
        model, opt, scheduler, start_step_count = load_model_opt_scheduler_states_fsdp(
            model, opt, scheduler, args.checkpoint_path)
    else:
        model = load_state_dict_fsdp(model, args.init_checkpoint_path)
        start_step_count = 0

    if args.act_checkpointing:

        def check_fn(submodule):
            return isinstance(submodule, wrapped_class)

        apply_activation_checkpointing(model, check_fn=check_fn)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_config_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    add_padding_token(tokenizer)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_path=args.data_path,
        data_fraction=args.data_fraction,
        seed=args.sample_seed,
    )
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    dataloader_full, sampler = get_dataloader_and_sampler(
        train_dataset=train_dataset,
        data_collator=data_collator,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )

    # updating the dataloader to the right state
    step_count = start_step_count
    sub_step_count = step_count * args.accumulation_steps
    start_epoch = sub_step_count // len(dataloader_full)
    skip_steps = sub_step_count % len(dataloader_full)
    sampler.set_epoch(start_epoch)
    dataloader = skip_first_batches(dataloader_full, skip_steps)
    print(
        "start_step_count",
        start_step_count,
        "step_count",
        step_count,
        "epoch",
        start_epoch,
        "skip_steps",
        skip_steps,
    )

    accumulation_steps = args.accumulation_steps
    sam_accumulation_steps = args.sam_accumulation_steps
    save_steps = args.save_steps
    epoch_iterator = iter(dataloader)
    start_time = time.time()
    for step_count in range(start_step_count, args.max_steps):
        train_loss = 0
        if args.sam_rho > 0:
            with torch.no_grad():
                w_t = {k: v.clone() for k, v in model.named_parameters()}
            for _ in range(4):
                try:
                    data = next(epoch_iterator)
                except StopIteration:
                    sampler.set_epoch(sampler.epoch + 1)
                    dataloader = dataloader_full
                    epoch_iterator = iter(dataloader)
                    data = next(epoch_iterator)
                out = model(**data)
                (out.loss / sam_accumulation_steps).backward()

            model.clip_grad_norm_(args.max_grad_norm)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data.add_(args.sam_rho * p.grad /
                                    (torch.norm(p.grad) + 1e-12))
            model.zero_grad()

        for _ in range(accumulation_steps):
            try:
                data = next(epoch_iterator)
            except StopIteration:
                sampler.set_epoch(sampler.epoch + 1)
                dataloader = dataloader_full
                epoch_iterator = iter(dataloader)
                data = next(epoch_iterator)
            # if args.replace_WN:
            #     # convert input_ids back to words
            #     input_ids = data["input_ids"]
            #     input_ids = input_ids.cpu().numpy()
            #     input_ids = tokenizer.batch_decode(input_ids,
            #                                        skip_special_tokens=True)
            #     t = Wordnet()
            #     input_ids = t.augment(input_ids)
            #     input_ids = tokenizer(input_ids,
            #                           padding="max_length",
            #                           truncation=True,
            #                           max_length=512,
            #                           return_tensors="pt")
            #     data["input_ids"] = input_ids["input_ids"].to(
            #         data["input_ids"].device)

            if args.drop_prob > 0:
                batch_size = data["input_ids"].shape[0]
                for i in range(batch_size):
                    data["input_ids"][i,
                                      torch.rand(data["input_ids"].shape[1]) <
                                      args.drop_prob] = 0
            if args.cutout_length > 0:
                batch_size = data["input_ids"].shape[0]
                # for each example, randomly choose a start position, plus a length, and set the input_ids to be 0
                # this is equivalent to cutout
                # we do this for each example in the batch
                input_lengths = torch.sum(data["attention_mask"], 1)  # B
                # cutout_start is #B , between 0 and input_lengths - cutout_length
                cutout_length = args.cutout_length * input_lengths
                cutout_length = cutout_length.long()
                start_pos = torch.zeros((batch_size, ), dtype=torch.long)
                for i in range(batch_size):
                    start_pos[i] = torch.randint(
                        0,
                        input_lengths[i] - cutout_length[i] + 1,
                        (1, ),
                    )
                for i in range(batch_size):
                    data["input_ids"][i, start_pos[i]:start_pos[i] +
                                      cutout_length[i]] = 0
            if args.mixup_strat is not None:
                if args.mixup_strat not in ["simple"]:
                    raise ValueError(
                        f"Unknown mixup strategy {args.mixup_strat}")
                if args.mixup_strat == "simple" and isinstance(
                        model,
                        torch.distributed.fsdp.fully_sharded_data_parallel.
                        FullyShardedDataParallel,
                ):
                    # First get embeddings
                    embed_device = (model._fsdp_wrapped_module.model.
                                    embed_tokens.weight.device)
                    embeds_init = model._fsdp_wrapped_module.model.embed_tokens.forward(
                        data["input_ids"].to(embed_device))
                    # Create mixup
                    # in the simple strategy, assume we have 2*n examples in the batch and n weights (w1, w2, ..., wn > 0.5), we what to create a new batch of embeddings as follows:
                    # First n examples: w1 * embeds_1 + (1-w1) * embeds_(n+1), w2 * embeds_2 + (1-w2) * embeds_(n+2), ..., wn * embeds_n + (1-wn) * embeds_2n
                    # Last n examples: (1-w1) * embeds_1 + w1 * embeds_(n+1), ..., (1-wn) * embeds_n + wn * embeds_2n
                    # Which means first n examples are dominated by its own embeddings, and last n examples are dominated by the embeddings of its own too, and we don't change the corresponding labels
                    # Beta distribution with mixup_alpha
                    batch_size = embeds_init.shape[0]
                    half_len = batch_size // 2
                    mixup_alpha = args.mixup_alpha
                    # Sample mixup weights for half the batch
                    mixup_weights_full = (torch.distributions.beta.Beta(
                        mixup_alpha, mixup_alpha).sample(
                            (batch_size, )).to(embeds_init))

                    # Ensure weights are greater than 0.5
                    mixup_weights_full = torch.where(
                        mixup_weights_full <= 0.5,
                        1 - mixup_weights_full,
                        mixup_weights_full,
                    )

                    # then, make weight <= 0.5 to be 1-weight
                    mixup_weights_full = mixup_weights_full.unsqueeze(
                        1).unsqueeze(2)
                    mixup_weights_full = mixup_weights_full.expand(
                        -1, embeds_init.shape[1], embeds_init.shape[2])
                    is_mixup = torch.rand((batch_size, )) < args.mixup_prob

                    reversed_idx = torch.arange(batch_size - 1,
                                                -1,
                                                -1,
                                                device=embeds_init.device)
                    embeds_to_mix = embeds_init[reversed_idx]

                    # Now we have the weights, we can create the new embeddings
                    # First n examples
                    embed_new = torch.zeros_like(embeds_init).to(
                        embeds_init)  # B x L x D
                    if args.mixup_detach:
                        embed_new = mixup_weights_full * embeds_init + (
                            (1 - mixup_weights_full) * embeds_to_mix).detach()
                    else:
                        embed_new = mixup_weights_full * embeds_init + (
                            1 - mixup_weights_full) * embeds_to_mix

                    embed_new[~is_mixup] = embeds_init[~is_mixup]

                    # embed_new = embed_new.detach()
                    data["inputs_embeds"] = embed_new
                    data["input_ids"] = None

            if args.neftune_alpha is not None:
                if isinstance(
                        model,
                        torch.distributed.fsdp.fully_sharded_data_parallel.
                        FullyShardedDataParallel,
                ):
                    embed_device = model._fsdp_wrapped_module.model.embed_tokens.weight.device

                    if data["input_ids"] is None:
                        embeds_init = data["inputs_embeds"].to(embed_device)
                    else:
                        embeds_init = model._fsdp_wrapped_module.model.embed_tokens.forward(
                            data["input_ids"].to(embed_device))

                    # add noise to embeds
                    input_mask = data["attention_mask"].to(
                        embeds_init)  # B x L
                    input_lengths = torch.sum(input_mask, 1)  # B

                    noise_ = torch.zeros_like(embeds_init).uniform_(-1, 1)
                    delta = noise_ * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.neftune_alpha / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
                    data["inputs_embeds"] = delta + embeds_init
                    data["input_ids"] = None
                    # add noise to embeds

            out = model(**data)

            (out.loss / accumulation_steps).backward()
            train_loss += out.loss.item() / accumulation_steps
        model.clip_grad_norm_(args.max_grad_norm)
        if rank == 0:
            time_so_far = (time.time() - start_time) / 3600
            iteration_so_far = step_count - start_step_count
            remaining_iterations = args.max_steps - step_count
            estimated_time_per_iteration = time_so_far / (iteration_so_far + 1)
            remaining_time = estimated_time_per_iteration * remaining_iterations
            previous_time = start_step_count * estimated_time_per_iteration
            total_estimated_time = time_so_far + remaining_time + previous_time
            metrics_dict = {
                "train/loss":
                train_loss,
                "train/learning_rate":
                scheduler.get_last_lr()[0],
                "train/global_step":
                step_count + 1,
                "train/time_so_far":
                time_so_far,
                "train/remaining_time":
                remaining_time,
                "train/total_estimated_time":
                total_estimated_time,
                "train/train_steps_per_second":
                1 / (estimated_time_per_iteration * 3600),
                "train/epoch":
                sampler.epoch,
            }
            if args.wandb:
                wandb.log(metrics_dict, step=step_count)
            print(json.dumps(metrics_dict, indent=4))

        if args.sam_rho > 0:
            with torch.no_grad():
                for p, w in zip(model.parameters(), w_t.values()):
                    p.copy_(w)  # copy back the original weights

        opt.step()
        scheduler.step()
        opt.zero_grad()

        # save the model, optimizer, scheduler
        if (step_count + 1) % save_steps == 0 or (step_count +
                                                  1) == args.max_steps:
            if rank == 0:
                print("saving checkpoint", step_count + 1)
            save_model_opt_scheduler_states_fsdp(
                model,
                opt,
                scheduler,
                step_count,
                args.checkpoint_path,
                rank,
                dont_save_opt=args.dont_save_opt,
                keep_old_checkpoints=args.keep_old_checkpoints,
            )

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path",
                        type=str,
                        default="llama/7B_sharded")
    parser.add_argument("--model_config_path", type=str, default="llama/7B_hf")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        default="llama/7B_checkpoint")
    parser.add_argument(
        "--wrapped_class_name",
        type=str,
        choices=["LlamaDecoderLayer", "OPTDecoderLayer"],
        default="LlamaDecoderLayer",
        help="the name of the class that is wrapped by the FSDP module",
    )
    parser.add_argument(
        "--dont_save_opt",
        action="store_true",
        help=
        "dont save optimizer and scheduler, this saves hard disk memory by trading off ability to resume the run",
    )
    parser.add_argument(
        "--keep_old_checkpoints",
        action="store_true",
        help="keep the intermediate checkpoints during training",
    )
    parser.add_argument("--added_tokens", type=int, default=1)
    parser.add_argument("--port", default=None)
    parser.add_argument("--data_path",
                        type=str,
                        default="data_instruct/alpaca.json")
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=1.0,
        help="fraction of data to use for training should be between 1 and 0",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="the random seed used for sampling a fraction of the data",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_steps", type=int, default=52002 * 3 // 128)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--hack",
        action="store_true",
        help=
        "This is a hack to reduce memory usage of the model by first casting the model to bf16 before moving to gpu"
        ", it uses less memory. However, it does not necessarily have the same training behavior as non-hacked version",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--act_checkpointing", action="store_true")
    parser.add_argument("--save_steps",
                        type=int,
                        default=(52002 * 3 / 128) // 10)
    parser.add_argument("--accumulation_steps", type=int, default=32)
    parser.add_argument("--neftune_alpha", type=float, default=None)

    # wandb associated arguments
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wb_project", type=str, default="data_instruct")
    parser.add_argument("--wb_name", type=str, default="test")

    # mixup arguments
    parser.add_argument("--mixup_strat", type=str, default=None)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--mixup_detach", action="store_true")

    # augmentation arguments
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--cutout_length", type=float, default=0)
    parser.add_argument("--replace_WN", action="store_true")
    parser.add_argument("--sam_rho", type=float, default=0.0)
    parser.add_argument("--sam_accumulation_steps", type=int, default=4)
    # train_option: 0: train all, 1: train only mlp layers, 2: train only self_attn layers, 3: train first half of the network, 4: train last half of the network
    parser.add_argument("--train_option", type=int, default=0)

    args = parser.parse_args()

    WORLD_SIZE = torch.cuda.device_count()
    if args.port is None:
        args.port = str(random.randint(
            1024, 65353))  # randomly generate ports if not specified
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
