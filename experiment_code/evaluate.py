import argparse
from ast import arg
import json
import os
import random
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.trainer_utils import seed_worker
import wandb

from dataset import make_supervised_data_module
from utils import (
    add_padding_token,
    load_fsdp_ckpt_with_accelerate,
)


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


def prepare_model(args):
    model_config = transformers.AutoConfig.from_pretrained(
        args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1
    final_ckpt_path = os.path.join(args.final_checkpoint_path, args.model_name,
                                   str(args.checkpoint_index), 'model')
    init_model = load_fsdp_ckpt_with_accelerate(
        args.init_checkpoint_path,
        model_config,
        hf_dummy_path=args.model_config_path,
        wrapped_class=args.wrapped_class_name,
    )
    final_model = load_fsdp_ckpt_with_accelerate(
        final_ckpt_path,
        model_config,
        hf_dummy_path=args.model_config_path,
        wrapped_class=args.wrapped_class_name,
    )
    return init_model, final_model


def prepare_dataset(args):
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
        rank=0,
        world_size=1,
    )
    return dataloader_full, sampler


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint_path",
                        type=str,
                        default="llama/7B_sharded")
    parser.add_argument("--model_config_path", type=str, default="llama/7B_hf")
    parser.add_argument("--final_checkpoint_path",
                        type=str,
                        default="llama/7B_checkpoint")
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--checkpoint_index", default=1217, type=int)
    parser.add_argument("--data_path",
                        type=str,
                        default="data_instruct/alpaca.json")
    parser.add_argument(
        "--data_fraction",
        type=float,
        default=0.1,
        help="fraction of data to use for training should be between 1 and 0",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="the random seed used for sampling a fraction of the data",
    )
    parser.add_argument(
        "--hack",
        action="store_true",
        help=
        "This is a hack to reduce memory usage of the model by first casting the model to bf16 before moving to gpu"
        ", it uses less memory. However, it does not necessarily have the same training behavior as non-hacked version",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--wrapped_class_name",
        type=str,
        choices=["LlamaDecoderLayer", "OPTDecoderLayer"],
        default="LlamaDecoderLayer",
        help="the name of the class that is wrapped by the FSDP module",
    )
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args


def compute_layerwise_distance(init_model, final_model):
    distances = {}
    for (init_name,
         init_param), (_, final_param) in zip(init_model.named_parameters(),
                                              final_model.named_parameters()):
        distance = torch.norm(init_param - final_param).item()
        distances[init_name] = distance
    return distances


def evaluate_model(model, dataloader):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            outputs = model(**batch)
            loss = outputs.loss.item()
            losses.append(loss)
            # progress bar
            if i % 100 == 0:
                print(f"evaluating {i} batches")
    return losses


if __name__ == "__main__":
    args = parse_arguments()
    result = {}
    dataloader_full, sampler = prepare_dataset(args)
    init_model, final_model = prepare_model(args)

    distance = compute_layerwise_distance(init_model, final_model)
    result["distance"] = distance

    print("model device", init_model.device)
    loss = evaluate_model(final_model, dataloader_full)
    result["loss"] = loss
    result["sample_seed"] = args.sample_seed
    result["data_fraction"] = args.data_fraction
    result["dataset"] = args.data_path.split("/")[-1].split(".")[0]

    data_name = args.data_path.split("/")[-1].split(".")[0]

    os.makedirs(f"./outputs/statistics", exist_ok=True)
    with open(
            f"./outputs/statistics/{args.model_name}_{args.checkpoint_index}_{data_name}.json",
            "w") as f:
        json.dump(result, f, indent=4)
