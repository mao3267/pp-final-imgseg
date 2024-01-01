from transformers import AutoImageProcessor, SegformerModel
import torch
from PIL import Image
import time
import multiprocessing as mp

def process_segment(args):
    filename = args
    image = Image.open(filename)
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b5")
    model = SegformerModel.from_pretrained("nvidia/mit-b5")
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs


if __name__ == "__main__":
    filename = "test_image.jpeg"
    pnum = 4
    
    start = time.time()
    
    pool = mp.Pool(processes=pnum)
    segment_args = [(filename) for i in range(pnum)]
    processed_segments = pool.map(process_segment, segment_args)

    # Wait for all the processes to complete
    pool.close()
    pool.join()
    
    end = time.time()

    print(f"Time taken: {end - start}")