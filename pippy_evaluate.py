# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch
import pippy

import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
)
from PIL import Image
from datasets import load_dataset
import evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.palette import ade_palette
import json
from huggingface_hub import cached_download, hf_hub_url
import time

pippy.fx.Tracer.proxy_buffer_attributes = True

gigabyte_size = 1024**3

metric = evaluate.load("mean_iou")

# Load label map
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(
    open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)


def format_to_gb(item, precision=4):
    """quick function to format numbers to gigabyte and round to (default) 4 digit precision"""
    metric_num = item / gigabyte_size
    metric_num = round(metric_num, ndigits=precision)
    return metric_num


def print_mem_usage():
    memory_reserved = format_to_gb(torch.cuda.memory_reserved())
    memory_allocated = format_to_gb(torch.cuda.memory_allocated())
    print(
        f"memory_reserved: {memory_reserved} GB, "
        f"memory_allocated: {memory_allocated} GB"
    )


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_all(pp_ranks, args):
    model = args.model
    model.eval()
    model.config.use_cache = False  # don't output `past_key_values`
    num_ranks = len(pp_ranks)

    if args.rank == 0:
        print(model.config)
        print(
            f"model total number of params = {get_number_of_params(model) // 10 ** 6}M"
        )

    split_policy = pippy.split_into_equal_size(num_ranks)

    # Use default value for kwargs other than `input_ids`
    concrete_args = pippy.create_default_args(
        model,
        # except_keys="input_ids",
    )

    pipe_driver, stage_mod = pippy.all_compile(
        model,
        num_ranks,
        args.chunks,
        split_policy=split_policy,
        tracer=PiPPyHFTracer(),
        concrete_args=concrete_args,
        index_filename=args.index_filename,
        # checkpoint_prefix=args.checkpoint_prefix,
    )

    params = get_number_of_params(stage_mod)
    print(f"submod_{args.rank} {params // 10 ** 6}M params")

    if args.rank != 0:
        return

    # Master continues
    print_mem_usage()

    # Inject pipeline driver's forward function back to original model to support HF's `generate()` method
    inject_pipeline_forward(model, pipe_driver)

    # Generate Image semantic segmentation results
    ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

    # Load feature extractor
    image_processor = SegformerImageProcessor.from_pretrained(
        args.model_name, reduce_labels=False
    )
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        args.model_name, reduce_labels=False
    )

    model.eval()

    mean_iou = 0.0

    mean_time = 0.0
    for i in range(0, len(ds), 2):
        image = Image.open(ds[i]["file"])
        segmentation_map = Image.open(ds[i + 1]["file"])

        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(
            args.device
        )
        start = time.time()
        outputs = model(pixel_values)
        end = time.time()
        mean_time += end - start

        labels = np.array(segmentation_map)

        logits = nn.functional.interpolate(
            outputs.logits.detach().cpu(),
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        seg = logits.argmax(dim=1)[0]
        color_seg = np.zeros(
            (seg.shape[0], seg.shape[1], 3), dtype=np.uint8
        )  # height, width, 3
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # Convert to BGR
        color_seg = color_seg[..., ::-1]

        # Show image + mask
        img = np.array(image) * 0.5 + color_seg * 0.5
        img = img.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        # plt.show()

        ground_truth_seg = np.array(
            segmentation_map
        )  # 2D ground truth segmentation map
        ground_truth_color_seg = np.zeros(
            (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), dtype=np.uint8
        )  # height, width, 3
        for label, color in enumerate(palette):
            ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
        # Convert to BGR
        ground_truth_color_seg = ground_truth_color_seg[..., ::-1]

        img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        img = img.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        # plt.show()

        pred_labels = logits.detach().cpu().numpy().argmax(axis=1)[0]

        # Compute metrics
        metrics = metric.compute(
            predictions=[
                pred_labels + 1
            ],  # add 1 to match the label ids in ground truth
            references=[labels],
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()

        # print(metrics)
        mean_iou += metrics["mean_iou"]

    print(f"mean_iou = {mean_iou / (len(ds) // 2)}")
    print(f"mean_time = {mean_time / (len(ds) // 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
    )
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument(
        "--master_addr", type=str, default=os.getenv("MASTER_ADDR", "localhost")
    )
    parser.add_argument(
        "--master_port", type=str, default=os.getenv("MASTER_PORT", "29500")
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nvidia/segformer-b5-finetuned-ade-640-640",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument(
        "--pp_group_size", type=int, default=int(os.getenv("WORLD_SIZE", 1))
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"]
    )
    parser.add_argument(
        "--index_filename",
        type=str,
        default=None,
        help="The director of model's index.json file",
    )
    # parser.add_argument(
    #     "--checkpoint_prefix",
    #     type=str,
    #     default=None,
    #     help="Prefix to add to the weight names in checkpoint map back to model structure",
    # )

    args = parser.parse_args()

    assert args.world_size % args.pp_group_size == 0

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        # Using float32 as default dtype to correspond to the default "fp32"
        # value for "--dtype"
        print(
            f"Unsupported data type {args.dtype}, "
            "please submit a PR to support it. Falling back to fp32 now."
        )
        dtype = torch.float32

    # Main process loads model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name, id2label=id2label, label2id=label2id
    )

    args.model = model
    args.gspmd = 1
    run_pippy(run_all, args)
