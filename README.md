# PaSSPI: Parallel Semantic Segmentation with Pipeline and Image Slicing

## Install

- Pippy

```
git clone https://github.com/pytorch/PiPPy.git
```

- Install Environment

```
pip install -r requirements.txt
```

## Run

### Run Segformer with PiPPy

```
python3 hf_generate.py
# measure latency
torchrun --nproc_per_node 8 pippy_evaluate.py
```

Current Model: `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`

### Slicing Images

#### Parallel Processing of Divided Photos

Divide the photo into four equal parts and sixteen equal parts, then utilize multi-threading for parallelization to process them simultaneously.

```
python3 slicing_evaluate.py
```

#### Parallel Processing of Multiple Photos

Utilize both multi-thread and multi-process to concurrently process multiple images, enhancing overall throughput and resource utilization.

- multi-thread

```
python3 multi_images_handling/test_threading.py
```

- multi-process

```
python3 multi_images_handling/test_multiprocessing.py
```

## Usage of cityscape dataset

1. unzip the gtFine_trainvaltest.zip file
2. only use aachen city now for simplicity

## Parallel GPU inference

To run `parallel_gpu.py` with multiple GPUs, use the command:

```
accelerate config
```

Follow the settings in the terminal to finish editing the config file.

After that, run the code by:

```
accelerate launch parallel_gpu.py
```
