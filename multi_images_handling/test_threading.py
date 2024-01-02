from transformers import AutoImageProcessor, SegformerModel
import torch
from PIL import Image
import time
import threading

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
    tnum = 4
    
    start = time.time()
    
    threads = []
    for i in range(tnum):
        thread = threading.Thread(target=process_segment, args=(filename,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    end = time.time()

    print(f"Time taken: {end - start}")