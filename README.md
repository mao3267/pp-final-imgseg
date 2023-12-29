# PaSSPI: Parallel Semantic Segmentation with Pipeline and Image Slicing

## Run Segformer with PiPPy

1. Install PiPPy: https://github.com/pytorch/PiPPy.git
2. Install requirements

```
pip install -r requierments.txt
```

3. Run in terminal

```
python3 hf_generate.py
# measure latency
torchrun --nproc_per_node 8 pippy_evaluate.py
```

Current Model: `nvidia/segformer-b1-finetuned-cityscapes-1024-1024`
