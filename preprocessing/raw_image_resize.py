import os
import cv2
import numpy as np
from glob import glob


def resize_with_padding(image, target_size=(1024, 768)):
    
    """
    Resize image to fit within target_size while maintaining aspect ratio.
    Pads with black (zero) pixels to reach the target dimensions.

    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded

def process_directory(input_dir, output_dir, target_size=(1024, 768)):

    """
    Process all .JPG files in input_dir, resize and pad them to target_size,
    and save the results as .png in output_dir.

    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob(os.path.join(input_dir, "*.JPG"))

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable file: {path}")
            continue

        padded_img = resize_with_padding(img, target_size)

        base_filename = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(output_dir, base_filename + ".png")
        cv2.imwrite(output_path, padded_img)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    input_directory = "dataset/Other_training/raw_images/"
    output_directory = "dataset/Other_training/nm_images/"
    process_directory(input_directory, output_directory)
