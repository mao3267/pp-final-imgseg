import evaluate
import torch.nn as nn

# import matplotlib.pyplot as plt
import numpy as np
from utils.palette import ade_palette
import json
from huggingface_hub import cached_download, hf_hub_url
import time
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
)
from PIL import Image
from pathlib import Path
from utils.cityscapes_id import trainId2label
from accelerate import Accelerator
import torch
from accelerate.utils import gather_object

accelerator = Accelerator()

image_to_process = 100


# Load label map
repo_id = "huggingface/label-files"
filename = "cityscapes-id2label.json"
id2label = json.load(
    open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
model.to(accelerator.device)

metric = evaluate.load("mean_iou")


# Load feature extractor
image_processor = SegformerImageProcessor.from_pretrained(
    model_name, reduce_labels=False
)
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    model_name, reduce_labels=False
)

count = 0

city_dir = Path("dataset/leftImg8bit/val")
files = city_dir.rglob("*_leftImg8bit.png")

color_image_dir = Path("dataset/gtFine/val")
color_files = color_image_dir.rglob("*_gtFine_labelIds.png")

# Filter out the relevant files for images and labels
image_files = [str(f) for f in files]  # if f.endswith("_leftImg8bit.png")
label_files = [str(f) for f in color_files]  # if f.endswith("_gtFine_labelIds.png")

# Sort the files to ensure matching pairs are aligned
image_files.sort()
label_files.sort()

image_files = image_files[:image_to_process]
label_files = label_files[:image_to_process]

accelerator.wait_for_everyone()
start = time.time()

zip_files = [(image_files[i], label_files[i]) for i in range(len(image_files))]

with accelerator.split_between_processes(zip_files) as inputs:
    results = dict(pred=[], label=[])

    for img_path, label_path in inputs:
        # Load the image and label
        local_start = time.time()
        image = Image.open(img_path)
        image = image.convert("RGB")
        label = Image.open(label_path)
        print(
            f"From GPU {accelerator.process_index}: Time taken to load image = {time.time() - local_start} seconds"
        )
        segmentation_map = np.array(label)

        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(
            accelerator.device
        )
        local_start = time.time()
        outputs = model(pixel_values)
        print(
            f"From GPU {accelerator.process_index}: Time taken for inference = {time.time() - local_start} seconds"
        )
        labels = np.array(segmentation_map)  # 2D ground truth segmentation map

        logits = nn.functional.interpolate(
            outputs.logits,  # .detach().cpu(),
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        # print(logits.shape)
        pred_labels = logits.argmax(dim=1)[0]  # .detach().cpu().numpy()
        count += 1

        results["pred"].append(pred_labels)
        results["label"].append(torch.tensor(labels))

    timediff = time.time() - start
    print(
        f"From GPU {accelerator.process_index}: Time taken for {len(inputs)} images = {timediff} seconds, {timediff / len(inputs)} seconds per image"
    )

    results = [results]


accelerator.wait_for_everyone()
results_gathered = gather_object(results)
print(f"\nFrom GPU {accelerator.process_index}: Gathered results")
print(f"time taken before metrics calculation = {time.time() - start} seconds")
if accelerator.is_main_process:
    mean_iou = 0.0
    mean_time = 0.0
    # print(type(results_gathered[0]))
    # print(len(results_gathered[0]["pred"]), len(results_gathered[0]["label"]))

    all_pred_labels = [r["pred"] for r in results_gathered]
    all_pred_labels = [item for sublist in all_pred_labels for item in sublist]
    all_labels = [r["label"] for r in results_gathered]
    all_labels = [item for sublist in all_labels for item in sublist]
    for index, pred_labels in enumerate(all_pred_labels):
        # print(f"From GPU {accelerator.process_index}: Start label conversion")

        pred_labels = pred_labels.detach().cpu().numpy()

        local_start = time.time()
        for i in range(pred_labels.shape[0]):
            for j in range(pred_labels.shape[1]):
                pred_labels[i][j] = trainId2label[pred_labels[i][j]][1]
        # print(
        #     f"From GPU {accelerator.process_index}: Time taken for label conversion = {time.time() - local_start} seconds"
        # )
        # print(f"From GPU {accelerator.process_index}: Start metric computation")
        # Compute metrics
        local_start = time.time()
        metrics = metric.compute(
            predictions=[pred_labels],  # add 1 to match the label ids in ground truth
            references=[all_labels[index].detach().cpu().numpy()],
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        # print(
        #     f"From GPU {accelerator.process_index}: Time taken for metric computation = {time.time() - local_start} seconds"
        # )
        # print(metrics)
        # print(f"miou: {metrics['mean_iou']}")
        mean_iou += metrics["mean_iou"]

    print(f"mean_iou = {mean_iou / image_to_process }")
    print(f"total time = {time.time() - start} seconds")
