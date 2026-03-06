"""
Data Preprocessing Script for Train_Roboflow Dataset
====================================================
This script:
1. Copies Train_Roboflow labeled images (already normalized and resized to 1024x768) as-is
2. Applies Reinhard normalization to Whole_Slides_Segments_Flattened unlabeled images
3. Resizes unlabeled images to 1024x768 with black padding using LANCZOS interpolation

Output Structure:
    dataset_processed2/
        labeled/
            images/     (from Train_Roboflow)
            labels/     (from Train_Roboflow .txt files)
        unlabeled/
            images/     (processed from Whole_Slides_Segments_Flattened)

Usage:
    python preprocess_data2.py

Requirements:
    - histomicstk
    - opencv-python
    - numpy
    - Pillow
    - tqdm
"""

import os
import cv2
import numpy as np
import glob
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import histomicstk as htk
    HISTOMICSTK_AVAILABLE = True
except ImportError:
    logger.error("histomicstk not found. Please install it using: pip install histomicstk")
    HISTOMICSTK_AVAILABLE = False
    sys.exit(1)


TARGET_SIZE = (1024, 768)
PADDING_COLOR = (0, 0, 0)

BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_raw")
OUTPUT_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code/dataset_processed2")

LABELED_IMG_DIR = BASE_DIR / "Train_Roboflow" / "train" / "images"
LABELED_LBL_DIR = BASE_DIR / "Train_Roboflow" / "train" / "labels"
UNLABELED_DIR = BASE_DIR / "Whole_Slides_Segments_Flattened"
REF_IMAGE_PATH = BASE_DIR / "color_reference.png"

TARGET_MEANS = np.array([8.74108109, -0.12440419,  0.0444982])
TARGET_STDS = np.array([0.6135447, 0.10989545, 0.10214027])

reference_mu_lab = None
reference_std_lab = None


def calculate_reference_statistics(ref_image_path):
    """Calculate LAB statistics from reference image using histomicstk method"""
    if not ref_image_path.exists():
        logger.error(f"Reference image not found: {ref_image_path}")
        logger.error("Reference image is required for Reinhard normalization!")
        sys.exit(1)
    
    try:
        ref_img = cv2.imread(str(ref_image_path))
        if ref_img is None:
            logger.error(f"Could not read reference image: {ref_image_path}")
            sys.exit(1)
        
        mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(ref_img)
        
        logger.info(f"Reference LAB means: {mean_ref}")
        logger.info(f"Reference LAB stds: {std_ref}")
        
        return mean_ref, std_ref
        
    except Exception as e:
        logger.error(f"Error calculating reference statistics: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def init_worker(ref_mu, ref_std):
    """Initialize worker process with reference statistics"""
    global reference_mu_lab, reference_std_lab
    reference_mu_lab = ref_mu
    reference_std_lab = ref_std


def resize_with_padding(image, size, color, method=Image.LANCZOS):
    """
    Resize image with aspect ratio preserved using padding.
    
    Args:
        image: PIL Image
        size: (width, height) tuple
        color: RGB tuple for padding color
        method: PIL resampling method
    
    Returns:
        PIL Image with padding
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


def apply_reinhard_normalization(img_bgr, mean_ref, std_ref):
    """
    Apply Reinhard color normalization using histomicstk.
    
    Args:
        img_bgr: numpy array (H, W, 3) in BGR format (from cv2.imread)
        mean_ref: reference LAB means from lab_mean_std
        std_ref: reference LAB stds from lab_mean_std
    
    Returns:
        Normalized numpy array in BGR format
    """
    try:
        im_nmzd = htk.preprocessing.color_normalization.reinhard(img_bgr, mean_ref, std_ref)
        return im_nmzd
    except Exception as e:
        logger.warning(f"Reinhard normalization failed: {e}, returning original")
        return img_bgr


def process_unlabeled_image(img_path):
    """
    Process a single unlabeled image:
    1. Read image (BGR format from cv2)
    2. Apply Reinhard normalization
    3. Convert BGR to RGB for PIL
    4. Resize to 1024x768 with black padding using LANCZOS
    5. Save as PNG
    
    Args:
        img_path: Path to image file
    
    Returns:
        None on success, error message on failure
    """
    global reference_mu_lab, reference_std_lab
    
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return f"Error reading {img_path}"
        
        if reference_mu_lab is not None and reference_std_lab is not None:
            normalized_bgr = apply_reinhard_normalization(
                img_bgr, 
                reference_mu_lab, 
                reference_std_lab
            )
        else:
            normalized_bgr = img_bgr
        
        normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(normalized_rgb.astype(np.uint8))
        
        resized_img = resize_with_padding(pil_img, TARGET_SIZE, PADDING_COLOR, Image.LANCZOS)
        
        save_name = img_path.stem + ".png"
        save_path = OUTPUT_DIR / "unlabeled" / "images" / save_name
        resized_img.save(save_path)
        
        return None
        
    except Exception as e:
        return f"Error processing {img_path.name}: {str(e)}"


def copy_labeled_data():
    """
    Copy Train_Roboflow data (images and labels) to dataset_processed2/labeled/
    
    Train_Roboflow contains:
    - Images: .jpg format, 1024x768, already Reinhard normalized with padding
    - Labels: .txt files in YOLO format
    
    Note: Images will be kept in their original format (.jpg), but we ensure
    the label files match the image stem names.
    """
    logger.info("Copying Train_Roboflow labeled data...")
    
    output_img_dir = OUTPUT_DIR / "labeled" / "images"
    output_lbl_dir = OUTPUT_DIR / "labeled" / "labels"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        image_files.extend(LABELED_IMG_DIR.glob(ext))
    
    logger.info(f"Found {len(image_files)} labeled images in Train_Roboflow")
    
    copied_images = 0
    copied_labels = 0
    missing_labels = []
    format_counts = {}
    
    for img_path in tqdm(image_files, desc="Copying labeled data"):
        ext = img_path.suffix.lower()
        format_counts[ext] = format_counts.get(ext, 0) + 1
        
        shutil.copy2(img_path, output_img_dir / img_path.name)
        copied_images += 1
        
        label_path = LABELED_LBL_DIR / img_path.with_suffix('.txt').name
        if label_path.exists():
            shutil.copy2(label_path, output_lbl_dir / label_path.name)
            copied_labels += 1
        else:
            missing_labels.append(img_path.name)
    
    logger.info(f"✓ Copied {copied_images} images")
    logger.info(f"  Image formats: {format_counts}")
    logger.info(f"✓ Copied {copied_labels} label files")
    
    if missing_labels:
        logger.warning(f"⚠ {len(missing_labels)} images missing label files:")
        for name in missing_labels[:10]:
            logger.warning(f"  - {name}")
        if len(missing_labels) > 10:
            logger.warning(f"  ... and {len(missing_labels) - 10} more")
    
    if '.jpg' in format_counts or '.jpeg' in format_counts:
        logger.info("\n" + "="*80)
        logger.info("NOTE: Image Format Handling")
        logger.info("="*80)
        logger.info("Labeled images are in .jpg format (from Train_Roboflow)")
        logger.info("Unlabeled images will be saved as .png format after processing")
        logger.info("This is fine for YOLO training - it handles mixed formats")
        logger.info("="*80 + "\n")
    
    return copied_images, copied_labels


def process_unlabeled_data(ref_mu, ref_std):
    """
    Process unlabeled images from Whole_Slides_Segments_Flattened:
    1. Apply Reinhard normalization
    2. Resize to 1024x768 with black padding
    3. Save to dataset_processed2/unlabeled/
    """
    logger.info("Processing unlabeled images...")
    
    output_dir = OUTPUT_DIR / "unlabeled" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unlabeled_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        unlabeled_files.extend(UNLABELED_DIR.glob(ext))
    
    logger.info(f"Found {len(unlabeled_files)} unlabeled images")
    
    if len(unlabeled_files) == 0:
        logger.warning(f"No images found in {UNLABELED_DIR}")
        return
    
    max_workers = multiprocessing.cpu_count()
    logger.info(f"Processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(ref_mu, ref_std)
    ) as executor:
        results = list(tqdm(
            executor.map(process_unlabeled_image, unlabeled_files),
            total=len(unlabeled_files),
            desc="Processing unlabeled images"
        ))
    
    errors = [r for r in results if r is not None]
    if errors:
        logger.warning(f"\n⚠ Encountered {len(errors)} errors:")
        for e in errors[:10]:
            logger.warning(f"  {e}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")
    
    successful = len(unlabeled_files) - len(errors)
    logger.info(f"✓ Successfully processed {successful}/{len(unlabeled_files)} unlabeled images")



def main():
    """Main preprocessing pipeline"""
    
    logger.info("="*80)
    logger.info("DATA PREPROCESSING - TRAIN_ROBOFLOW DATASET")
    logger.info("="*80)
    logger.info(f"Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    logger.info(f"Padding color: Black {PADDING_COLOR}")
    logger.info(f"Interpolation: LANCZOS")
    logger.info(f"Normalization: Reinhard (histomicstk)")
    logger.info("="*80)
    
    if not HISTOMICSTK_AVAILABLE:
        logger.error("histomicstk is required but not installed!")
        return False
    
    if not LABELED_IMG_DIR.exists():
        logger.error(f"Labeled data directory not found: {LABELED_IMG_DIR}")
        return False
    
    if not UNLABELED_DIR.exists():
        logger.error(f"Unlabeled data directory not found: {UNLABELED_DIR}")
        return False
    
    logger.info("\nStep 1: Calculating reference statistics...")
    ref_mu, ref_std = calculate_reference_statistics(REF_IMAGE_PATH)
    
    logger.info("\nStep 2: Copying labeled data from Train_Roboflow...")
    num_images, num_labels = copy_labeled_data()
    
    logger.info("\nStep 3: Processing unlabeled images...")
    process_unlabeled_data(ref_mu, ref_std)
    
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Labeled images: {num_images}")
    logger.info(f"Label files: {num_labels}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*80)
    
    logger.info("\nNext steps:")
    logger.info("1. Run prepare_yolo_dataset2.py to split Train_Roboflow into train/val/test")
    logger.info("2. Use dataset_yolo2 for YOLO training")
    
    return True



if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
