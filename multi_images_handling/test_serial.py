from transformers import AutoImageProcessor, SegformerModel
import torch
from PIL import Image
import time

if __name__ == "__main__":
    filename = "test_image.jpeg"
    num = 4
    
    start = time.time()
    
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b5")
    model = SegformerModel.from_pretrained("nvidia/mit-b5")
    
    for i in range(num):
        image = Image.open(filename)

        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

    end = time.time()

    print(f"Time taken: {end - start}")