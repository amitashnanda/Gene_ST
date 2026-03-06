import os
import cv2
import numpy as np
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import sys

try:
    from tiatoolbox.tools.stainnorm import VahadaneNormalizer
    TIATOOLBOX_AVAILABLE = True
except ImportError:
    print("tiatoolbox not found. Please install it using: pip install tiatoolbox")
    TIATOOLBOX_AVAILABLE = False

TARGET_SIZE = (1024, 1024)

BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_raw")

NEW_TEST_DIR = BASE_DIR / "new_test"

OUTPUT_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_processed/new_test")

REF_IMAGE_PATH = BASE_DIR / "color_reference.png"

normalizer = None


def get_user_options():
    """
    Ask user for preprocessing options via CLI.
    """
    print("\n--- Preprocessing Configuration (new_test) ---")
    
    while True:
        resp = input("Do you want to use padding to preserve aspect ratio? (y/n) [default: n]: ").lower().strip()
        if resp in ['', 'n', 'no']:
            use_padding = False
            print("Selected: Direct resizing (stretching).")
            break
        elif resp in ['y', 'yes']:
            use_padding = True
            print("Selected: Resize with padding.")
            break
        else:
            print("Please answer y or n.")

    padding_color = (114, 114, 114)
    
    if use_padding:
        print("\nSelect padding color:")
        print("1. Grey (Standard YOLO - 114,114,114) [Default]")
        print("2. Black (0,0,0)")
        print("3. White (255,255,255)")
        
        while True:
            resp = input("Enter choice [1-3]: ").strip()
            if resp in ['', '1']:
                padding_color = (114, 114, 114)
                print("Selected: Grey padding.")
                break
            elif resp == '2':
                padding_color = (0, 0, 0)
                print("Selected: Black padding.")
                break
            elif resp == '3':
                padding_color = (255, 255, 255)
                print("Selected: White padding.")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
    return use_padding, padding_color

def init_worker(ref_img_path: Path):
    """Initialize the normalizer in each worker process."""
    global normalizer
    if not TIATOOLBOX_AVAILABLE:
        return
    
    if not ref_img_path.exists():
        return
    
    try:
        ref_img = cv2.imread(str(ref_img_path))
        if ref_img is not None:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            normalizer = VahadaneNormalizer()
            normalizer.fit(ref_img)
    except Exception:
        pass

def resize_with_padding(image: Image.Image, size, color, method=Image.LANCZOS):
    """
    Resize image with unchanged aspect ratio using padding.
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), method)
    new_image = Image.new('RGB', size, color)
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image

def resize_direct(image: Image.Image, size, method=Image.LANCZOS):
    """
    Resize image directly to target size without padding (stretching).
    """
    return image.resize(size, method)

def process_file(args):
    """
    Process a single file from new_test.
    args: (img_path, use_padding, padding_color)
    """
    img_path, use_padding, padding_color = args
    global normalizer
    
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return f"Error reading {img_path}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if normalizer is not None:
            try:
                norm_img = normalizer.transform(img)
            except Exception:
                norm_img = img
        else:
            norm_img = img

        pil_img = Image.fromarray(norm_img.astype(np.uint8))
        
        if use_padding:
            resized_img = resize_with_padding(pil_img, TARGET_SIZE, padding_color, Image.LANCZOS)
        else:
            resized_img = resize_direct(pil_img, TARGET_SIZE, Image.LANCZOS)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        save_name = img_path.stem + ".png"
        save_path = OUTPUT_DIR / save_name
        resized_img.save(save_path)

        return None

    except Exception as e:
        return f"Error processing {img_path.name}: {str(e)}"


def process_new_test_dataset():
    if not TIATOOLBOX_AVAILABLE:
        print("Cannot run: tiatoolbox is not installed.")
        return

    if not NEW_TEST_DIR.exists():
        print(f"Input folder not found: {NEW_TEST_DIR}")
        return

    if not REF_IMAGE_PATH.exists():
        print(f"Reference image not found at {REF_IMAGE_PATH}")
        return

    use_padding, padding_color = get_user_options()

    print("\nCollecting files from new_test...")
    img_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.tif", "*.tiff"]:
        img_files.extend(NEW_TEST_DIR.glob(ext))

    if not img_files:
        print(f"No images found in {NEW_TEST_DIR}")
        return

    print(f"Found {len(img_files)} images in new_test.")

    tasks = [(f, use_padding, padding_color) for f in img_files]

    max_workers = multiprocessing.cpu_count()
    print(f"Starting processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(REF_IMAGE_PATH,)
    ) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks)))

    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print("...")

    print(f"\nProcessing complete. Processed images saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_new_test_dataset()
