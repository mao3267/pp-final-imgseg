# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os

import torch
import pippy

import pippy.fx
from pippy import run_pippy
from pippy.hf import PiPPyHFTracer, inject_pipeline_forward
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    pipeline,
)
from PIL import Image
from datasets import load_dataset
import evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.palette import create_ade20k_label_colormap
import requests

pippy.fx.Tracer.proxy_buffer_attributes = True

gigabyte_size = 1024**3


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
        except_keys="input_ids",
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
    # ds = load_dataset("", split="test[:10]")

    # Load feature extractor
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    # Load image
    # image = ds[6]["image"]
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # print image
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    plt.show()

    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to("cpu")

    # Run inference
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(create_ade20k_label_colormap())
    for label, color in enumerate(palette):
        color_seg[pred_seg == label, :] = color
    color_seg = color_seg[..., ::-1]  # convert to BGR

    img = (
        np.array(image) * 0.5 + color_seg * 0.5
    )  # plot the image with the segmentation map
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world_size", type=int, default=int(os.getenv("WORLD_SIZE", 1))
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
        default="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
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
        args.model_name,
        num_labels=19,
    )

    args.model = model
    args.gspmd = 1
    run_pippy(run_all, args)
