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

# from datasets import load_dataset
import evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils.palette import ade_palette

# from utils.cityscapes_utils import CityscapesLabelEncoder, CityscapesTrainDataset, CityscapesTestDataset, CityscapesDataset
import json
from huggingface_hub import cached_download, hf_hub_url
import time

# from measure_latency import measure_latency
import warnings
from torchvision.datasets import Cityscapes
from pathlib import Path
from utils.cityscapes_id import trainId2label

warnings.filterwarnings("ignore", message=".*is deprecated.*")
# Suppress all warnings
warnings.filterwarnings("ignore")

pippy.fx.Tracer.proxy_buffer_attributes = True

gigabyte_size = 1024**3

metric = evaluate.load("mean_iou")

# Load label map
repo_id = "huggingface/label-files"
filename = "cityscapes-id2label.json"
id2label = json.load(
    open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(f"num_labels = {num_labels}")
pippy_latency = 0.0
image_to_process = 10


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
    print(f"device: {args.device}")
    model.to(torch.device(args.device))
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

    image_processor = SegformerImageProcessor.from_pretrained(
        args.model_name, reduce_labels=False
    )
    feature_extractor = SegformerFeatureExtractor.from_pretrained(
        args.model_name, reduce_labels=False
    )

    mean_iou = 0.0

    mean_time = 0.0

    # Dataset structure: ./dataset -> leftImg8bit_trainvaltest/gtFine_trainvaltest
    city_dir = Path("dataset/leftImg8bit/val")
    files = city_dir.rglob("*_leftImg8bit.png")

    color_image_dir = Path("dataset/gtFine/val")
    color_files = color_image_dir.rglob("*_gtFine_labelIds.png")

    # city_name = "aachen"
    # city_dir = os.path.join("dataset/leftImg8bit_trainvaltest/leftImg8bit/", city_name)
    # files = os.listdir(city_dir)

    # color_image_dir = os.path.join(
    #     "dataset/gtFine_trainvaltest/gtFine/train", city_name
    # )
    # color_files = os.listdir(color_image_dir)

    # Filter out the relevant files for images and labels
    image_files = [str(f) for f in files]  # if f.endswith("_leftImg8bit.png")
    label_files = [str(f) for f in color_files]  # if f.endswith("_gtFine_labelIds.png")

    # Sort the files to ensure matching pairs are aligned
    image_files.sort()
    label_files.sort()

    # Iterate over each image and label pair
    count = 0
    for img_path, label_path in zip(image_files, label_files):
        # Construct the full file paths

        # Check if the prefixes are the same
        # if img_prefix != lbl_prefix:
        #     print(f"Error: Mismatched image and label pair: {img_file}, {lbl_file}")
        #     continue
        # Load the image and label
        image = Image.open(img_path)
        image = image.convert("RGB")
        label = Image.open(label_path)
        # segmentation_map = label.convert("L")
        segmentation_map = np.array(label)

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
        image = Image.fromarray(img)
        image.save("results.jpg")
        # plt.figure(figsize=(15, 10))
        # plt.imshow(img)
        # plt.show()

        # ground_truth_seg = np.array(
        #     segmentation_map
        # )  # 2D ground truth segmentation map
        # ground_truth_color_seg = np.zeros(
        #     (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), dtype=np.uint8
        # )  # height, width, 3
        # for label, color in enumerate(palette):
        #     ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
        # # Convert to BGR
        # ground_truth_color_seg = ground_truth_color_seg[..., ::-1]

        # img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
        # img = img.astype(np.uint8)

        # plt.figure(figsize=(15, 10))
        # plt.imshow(img)
        # plt.show()

        pred_labels = logits.detach().cpu().numpy().argmax(axis=1)[0]

        # Convert trainId to labelId
        for i in range(pred_labels.shape[0]):
            for j in range(pred_labels.shape[1]):
                pred_labels[i][j] = trainId2label[pred_labels[i][j]][1]
        # with open("pred_labels.txt", "w") as f:
        #     for i in range(pred_labels.shape[0]):
        #         for j in range(pred_labels.shape[1]):
        #             f.write(str(pred_labels[i][j]) + " ")
        #         f.write("\n")
        # with open("labels.txt", "w") as f:
        #     for i in range(labels.shape[0]):
        #         for j in range(labels.shape[1]):
        #             f.write(str(labels[i][j]) + " ")
        #         f.write("\n")
        # Compute metrics

        metrics = metric.compute(
            predictions=[pred_labels],  # add 1 to match the label ids in ground truth
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
        print(f"iou: {metrics['mean_iou']}")
        count += 1
        if count == image_to_process:
            break

    print(f"mean_iou = {mean_iou / count }")
    print(f"mean_time = {mean_time / count }")
    print(f"all time = {mean_time}")
    global pippy_latency
    # pippy_latency = measure_latency(model=model, measure_times=10, num_threads=1, MODEL_NAME="Segformer on PiPPy")


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
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=int(torch.cuda.is_available()))
    parser.add_argument(
        "--pp_group_size", type=int, default=int(os.getenv("WORLD_SIZE", 4))
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
    # if args.rank == 0:
    # model_latency = measure_latency(model=model, measure_times=10, num_threads=1, MODEL_NAME=args.model_name)

    args.model = model
    args.gspmd = 1
    run_pippy(run_all, args)

    if args.rank != 0:
        exit()

    # print(f"--- Latency(sec) ---")
    # print("pippy latency = ", pippy_latency)
    # print("model latency = ", model_latency)
    # print(f"speedup = {model_latency / pippy_latency:.2f}")
