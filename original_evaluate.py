from datasets import load_dataset
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
import torch
from pathlib import Path
from utils.cityscapes_id import trainId2label

image_to_process = 10
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

model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model.to(device)
model.eval()

# Load image
# Generate Image semantic segmentation results
# ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

# Load feature extractor
image_processor = SegformerImageProcessor.from_pretrained(
    model_name, reduce_labels=False
)
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    model_name, reduce_labels=False
)

mean_iou = 0.0

mean_time = 0.0
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

    segmentation_map = np.array(label)

    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    start = time.time()
    outputs = model(pixel_values)
    end = time.time()
    mean_time += end - start

    labels = np.array(segmentation_map)  # 2D ground truth segmentation map

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

    # plt.figure(figsize=(15, 10))
    # plt.imshow(img)
    # plt.show()

    ground_truth_seg = np.array(segmentation_map)  # 2D ground truth segmentation map
    ground_truth_color_seg = np.zeros(
        (ground_truth_seg.shape[0], ground_truth_seg.shape[1], 3), dtype=np.uint8
    )  # height, width, 3
    for label, color in enumerate(palette):
        ground_truth_color_seg[ground_truth_seg - 1 == label, :] = color
    # Convert to BGR
    ground_truth_color_seg = ground_truth_color_seg[..., ::-1]

    img = np.array(image) * 0.5 + ground_truth_color_seg * 0.5
    img = img.astype(np.uint8)

    # plt.figure(figsize=(15, 10))
    # plt.imshow(img)
    # plt.show()

    pred_labels = logits.detach().cpu().numpy().argmax(axis=1)[0]

    for i in range(pred_labels.shape[0]):
        for j in range(pred_labels.shape[1]):
            pred_labels[i][j] = trainId2label[pred_labels[i][j]][1]

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
    print(f"iou = {metrics['mean_iou']}")
    count += 1

    if count == image_to_process:
        break

print(f"mean_iou = {mean_iou / count }")
print(f"mean_time = {mean_time / count }")
