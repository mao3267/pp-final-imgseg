from datasets import load_dataset
import evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
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

from threading import Thread

def process_tensor_segment(model, tensor_segment, results, index):
    # Assuming the model can process tensor directly
    output = model(tensor_segment)
    results[index] = output.logits


def process_image_without_overlap(model, pixel_values):
    _, _, width, height = pixel_values.shape
    
    segments = [
        pixel_values[:, :, :width // 2, :height // 2],
        pixel_values[:, :, width // 2:, :height // 2],
        pixel_values[:, :, :width // 2, height // 2:],
        pixel_values[:, :, width // 2:, height // 2:]
    ]
    
    threads = []
    results = [None] * 4
    for i, segment in enumerate(segments):
        thread = Thread(target=process_tensor_segment, args=(model, segment, results, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # Concatenate the results
    top_half = torch.cat((results[0], results[2]), dim=3)
    bottom_half = torch.cat((results[1], results[3]), dim=3)
    full_image = torch.cat((top_half, bottom_half), dim=2)

    return full_image
    

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

model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
model = SegformerForSemanticSegmentation.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

# Load image
# Generate Image semantic segmentation results
ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")

# Load feature extractor
image_processor = SegformerImageProcessor.from_pretrained(
    model_name, reduce_labels=False
)
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    model_name, reduce_labels=False
)

model.eval()

mean_iou = 0.0

mean_time = 0.0
for i in range(0, len(ds), 2):
    image = Image.open(ds[i]["file"])
    segmentation_map = Image.open(ds[i + 1]["file"])
    
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    start = time.time()
    
    # outputs = model(pixel_values)
    outputs = process_image_without_overlap(model, pixel_values)
    
    end = time.time()
    mean_time += end - start

    labels = np.array(segmentation_map)

    logits = nn.functional.interpolate(
        outputs.detach().cpu(),
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

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    # plt.show()

    pred_labels = logits.detach().cpu().numpy().argmax(axis=1)[0]

    # Compute metrics
    metrics = metric.compute(
        predictions=[pred_labels + 1],  # add 1 to match the label ids in ground truth
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
