import timm, torch
import numpy as np
from PIL import Image

import time
from transformers import SegformerForSemanticSegmentation

def measure_latency(
    MODEL_NAME,
    model, 
    measure_times=100, 
    warm_up=3, 
    num_threads=2,
    input_shape=(1, 3, 224, 224),
):
    torch.set_num_threads(num_threads)
    MEASURE_TIMES = measure_times
    WARM_UP = warm_up
    BATCH_SIZE = input_shape[0]
    
    print(f"\n\nMeasuring average inference latecy of {MODEL_NAME} over {MEASURE_TIMES} run.")
    print(f"  - batch size = {BATCH_SIZE}")
    print(f"  - nums_threads = {num_threads}")
    
    latency_list = []
    time.sleep(10)  # cool-down CPU for fair comparison
    with torch.no_grad():
        for i in range(MEASURE_TIMES + WARM_UP):            
            input = torch.randn(BATCH_SIZE,3,224,224)
            
            start_time = time.perf_counter()
            output = model(input)
            end_time = time.perf_counter()
            
            if(i<WARM_UP):  # ignore the first 3 runs
                continue
            
            inference_time = end_time - start_time
            latency_list.append(inference_time)
            print(f"Inference time for {MODEL_NAME}: iteration_{i - WARM_UP} = {inference_time:.3f} || average = {np.mean(latency_list):.3f} seconds ")
    # print(latency_list)
    return np.mean(latency_list)

if __name__ == "__main__":  
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
    )
    measure_latency(model=model, measure_times=10, num_threads=1, MODEL_NAME="nvidia/segformer-b5-finetuned-ade-640-640")
    