import struct
import numpy as np
from PIL import Image, ImageOps
import sys

def process_and_save(image_path, output_bin_path):
    print(f"Processing {image_path}...")
    
    # 1. Open and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # 2. Invert (MNIST digits are white on black background)
    # If your input image is already white on a black background, comment this out!
    aws = input("is the image white on black?(y/n)")
    if (aws.lower() == "n"):
        print("CONVERTING..")
        img = ImageOps.invert(img) 
    
    # 3. Resize to 28x28 exactly like MNIST
    img = img.resize((28, 28))
    
    # 4. Convert to Numpy array and normalize to [0.0, 1.0]
    img_array = np.array(img, dtype=np.float64) / 255.0
    flat_array = img_array.flatten()
    
    # 5. Pack exactly 784 doubles ('d') into a binary file
    with open(output_bin_path, 'wb') as f:
        f.write(struct.pack(f'{len(flat_array)}d', *flat_array))
    
    print(f"Saved 784 formatted doubles to {output_bin_path}!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_image.py my_digit.png input_image.bin")
    else:
        process_and_save(sys.argv[1], sys.argv[2])
