"""
YOLOv11 Segmentation Inference on Test Set with Enhanced Visualizations
=======================================================================
This script:
- Evaluates the trained model on the test set (mAP, precision, recall, etc.)
- Runs prediction on test images
- Saves detailed visualizations with bounding boxes and masks overlaid
- Generates comparison grids and individual annotated images

Usage:
    python 4_inference_test_set.py

Requirements:
    - ultralytics
    - torch with CUDA support
    - matplotlib
    - opencv-python
    - numpy
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("yolo11_inference")


class InferenceConfig:
    """Configuration for inference on test set"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")

    MODEL_PATH = BASE_DIR / "runs/yolo11m_seg_dinov3_20251208_093550/weights/last.pt"

    DATA_YAML = BASE_DIR / "dataset_yolo2/data.yaml"

    BATCH_SIZE = 16

    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    
    COLORS = {
        'crypt': (255, 0, 0),
        'gland': (0, 255, 0),
        'text': (255, 255, 255),
        'bbox': 2,
        'mask_alpha': 0.4,
    }
    
    CLASS_NAMES = ['crypt', 'gland']

    DEVICE = 0

    OUTPUT_DIR = BASE_DIR / "runs/yolo11_segmentation_inference"
    VIS_DIR = BASE_DIR / "visualizations/test_predictions"
    EXPERIMENT_NAME = f"yolo11m_seg_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def __post_init__(self):
        """Validate configuration and create directories"""
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {self.MODEL_PATH}")

        if not self.DATA_YAML.exists():
            raise FileNotFoundError(f"data.yaml not found at {self.DATA_YAML}")

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VIS_DIR.mkdir(parents=True, exist_ok=True)

        if not torch.cuda.is_available():
            logger.warning("CUDA not available; inference will run on CPU.")
            self.DEVICE = 'cpu'
        else:
            logger.info(f"Using device: {self.DEVICE}")

    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMAGE_SIZE = (768, 1024)
    DEVICE = 0
    SAVE_CONF = True



def overlay_masks_on_image(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, 
                          classes: np.ndarray, confidences: np.ndarray, 
                          config: InferenceConfig) -> np.ndarray:
    """
    Overlay segmentation masks and bounding boxes on the original image.
    
    Args:
        image: Original image (H, W, 3) in BGR format
        masks: Binary masks (N, H, W)
        boxes: Bounding boxes (N, 4) in xyxy format
        classes: Class indices (N,)
        confidences: Confidence scores (N,)
        config: Configuration object
    
    Returns:
        Annotated image in BGR format
    """
    annotated_img = image.copy()
    h, w = image.shape[:2]
    
    if masks is None or len(masks) == 0:
        logger.warning("No detections to visualize")
        return annotated_img
    
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    
    for idx, (mask, box, cls, conf) in enumerate(zip(masks, boxes, classes, confidences)):
        cls_idx = int(cls)
        class_name = config.CLASS_NAMES[cls_idx]
        color = config.COLORS[class_name]
        
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        mask_bool = mask > 0.5
        mask_overlay[mask_bool] = color
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, config.COLORS['bbox'])
        
        label = f"{class_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1] + 10)
        
        cv2.rectangle(annotated_img, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        
        cv2.putText(annotated_img, label, (x1 + 2, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS['text'], 2)
    
    annotated_img = cv2.addWeighted(annotated_img, 1.0, mask_overlay, 
                                    config.COLORS['mask_alpha'], 0)
    
    return annotated_img


def create_side_by_side_comparison(original: np.ndarray, annotated: np.ndarray, 
                                   image_name: str, save_path: Path):
    """
    Create side-by-side comparison of original and annotated images.
    
    Args:
        original: Original image (BGR)
        annotated: Annotated image (BGR)
        image_name: Name of the image
        save_path: Path to save the comparison
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Segmentation Results: {image_name}', fontsize=16, fontweight='bold')
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(annotated_rgb)
    axes[1].set_title('Predicted Segmentation (Boxes + Masks)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_grid_visualization(image_paths: List[Path], annotated_images: List[np.ndarray],
                             save_path: Path, grid_size: Tuple[int, int] = (4, 4)):
    """
    Create a grid of annotated images for quick overview.
    
    Args:
        image_paths: List of original image paths
        annotated_images: List of annotated images (BGR)
        save_path: Path to save the grid
        grid_size: Grid dimensions (rows, cols)
    """
    rows, cols = grid_size
    num_images = min(len(annotated_images), rows * cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.suptitle('Test Set Predictions Overview', fontsize=20, fontweight='bold')
    
    axes_flat = axes.flatten() if num_images > 1 else [axes]
    
    for idx in range(rows * cols):
        ax = axes_flat[idx]
        
        if idx < num_images:
            img_rgb = cv2.cvtColor(annotated_images[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(image_paths[idx].stem, fontsize=10)
        else:
            ax.axis('off')
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Grid visualization saved to: {save_path}")


def save_detection_statistics(results_list: List, save_path: Path, config: InferenceConfig):
    """
    Save statistics about detections across the test set.
    
    Args:
        results_list: List of YOLO results objects
        save_path: Path to save statistics
        config: Configuration object
    """
    stats = {
        'total_images': len(results_list),
        'images_with_detections': 0,
        'total_detections': 0,
        'detections_per_class': {name: 0 for name in config.CLASS_NAMES},
        'avg_confidence': 0.0,
        'confidence_per_class': {name: [] for name in config.CLASS_NAMES},
    }
    
    for result in results_list:
        if result.boxes is not None and len(result.boxes) > 0:
            stats['images_with_detections'] += 1
            stats['total_detections'] += len(result.boxes)
            
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = config.CLASS_NAMES[cls_idx]
                
                stats['detections_per_class'][class_name] += 1
                stats['confidence_per_class'][class_name].append(conf)
    
    for class_name in config.CLASS_NAMES:
        confs = stats['confidence_per_class'][class_name]
        if confs:
            stats['confidence_per_class'][class_name] = float(np.mean(confs))
        else:
            stats['confidence_per_class'][class_name] = 0.0
    
    all_confs = []
    for result in results_list:
        if result.boxes is not None and len(result.boxes) > 0:
            all_confs.extend([float(box.conf[0]) for box in result.boxes])
    
    if all_confs:
        stats['avg_confidence'] = float(np.mean(all_confs))
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST SET DETECTION STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total images: {stats['total_images']}\n")
        f.write(f"Images with detections: {stats['images_with_detections']}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Average confidence: {stats['avg_confidence']:.4f}\n\n")
        
        f.write("Detections per class:\n")
        for class_name in config.CLASS_NAMES:
            count = stats['detections_per_class'][class_name]
            avg_conf = stats['confidence_per_class'][class_name]
            f.write(f"  - {class_name}: {count} detections (avg conf: {avg_conf:.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"Detection statistics saved to: {save_path}")
    return stats



def resolve_test_images_dir(config: InferenceConfig) -> str | None:
    """
    Read data.yaml to find the test images directory.
    """
    with open(config.DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    test_entry = data.get("test", None)
    if test_entry is None:
        logger.warning("No 'test' entry in data.yaml.")
        return None

    test_path = Path(test_entry)

    if not test_path.is_absolute():
        test_path = (config.DATA_YAML.parent / test_path).resolve()

    if not test_path.exists():
        logger.warning(f"Test path does not exist: {test_path}")
        return None

    return str(test_path)


def run_inference_with_visualization(config: InferenceConfig, model: YOLO):
    """
    Run inference on test images and create comprehensive visualizations.
    """
    test_dir = resolve_test_images_dir(config)
    if test_dir is None:
        logger.error("Cannot find test directory. Skipping inference.")
        return
    
    test_path = Path(test_dir)
    logger.info(f"Running inference on test images: {test_path}")
    
    image_files = sorted(list(test_path.glob("*.png")) + list(test_path.glob("*.jpg")))
    
    if not image_files:
        logger.error(f"No images found in {test_path}")
        return
    
    logger.info(f"Found {len(image_files)} test images")
    
    annotated_dir = config.VIS_DIR / "annotated_images"
    comparison_dir = config.VIS_DIR / "side_by_side"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Running predictions...")
    results = model.predict(
        source=str(test_path),
        imgsz=config.IMAGE_SIZE,
        conf=config.CONF_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        device=config.DEVICE,
        save=False,
        verbose=False,
    )
    
    logger.info(f"Processing {len(results)} results...")
    
    annotated_images = []
    processed_paths = []
    
    for idx, (result, img_path) in enumerate(zip(results, image_files)):
        logger.info(f"Processing {idx + 1}/{len(results)}: {img_path.name}")
        
        original_img = cv2.imread(str(img_path))
        
        if original_img is None:
            logger.warning(f"Could not load image: {img_path}")
            continue
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
            else:
                masks = None
            
            annotated = overlay_masks_on_image(
                original_img, masks, boxes, classes, confidences, config
            )
            
            annotated_path = annotated_dir / f"{img_path.stem}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            
            comparison_path = comparison_dir / f"{img_path.stem}_comparison.png"
            create_side_by_side_comparison(original_img, annotated, img_path.name, comparison_path)
            
            annotated_images.append(annotated)
            processed_paths.append(img_path)
            
            logger.info(f"  ✓ Saved: {annotated_path.name} ({len(boxes)} detections)")
        else:
            annotated_path = annotated_dir / f"{img_path.stem}_annotated.png"
            cv2.imwrite(str(annotated_path), original_img)
            
            annotated_images.append(original_img)
            processed_paths.append(img_path)
            
            logger.info(f"  ✓ No detections found")
    
    if annotated_images:
        grid_path = config.VIS_DIR / "predictions_grid.png"
        create_grid_visualization(processed_paths, annotated_images, grid_path)
    
    stats_path = config.VIS_DIR / "detection_statistics.txt"
    save_detection_statistics(results, stats_path, config)
    
    logger.info(f"\n{'='*80}")
    logger.info("VISUALIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Annotated images: {annotated_dir}")
    logger.info(f"Side-by-side comparisons: {comparison_dir}")
    logger.info(f"Grid overview: {config.VIS_DIR / 'predictions_grid.png'}")
    logger.info(f"Statistics: {stats_path}")
    logger.info(f"{'='*80}\n")


def run_test_val(config: InferenceConfig, model: YOLO):
    """
    Run evaluation on the test split to get metrics.
    """
    logger.info("Running evaluation on test split...")

    results = model.val(
        data=str(config.DATA_YAML),
        split="test",
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        project=str(config.OUTPUT_DIR),
        name=config.EXPERIMENT_NAME + "_metrics",
        save_json=True,
        plots=True,
    )

    try:
        metrics_dict = results.results_dict
        logger.info("\n" + "="*80)
        logger.info("TEST SET METRICS")
        logger.info("="*80)
        for k, v in metrics_dict.items():
            logger.info(f"{k}: {v:.4f}")
        logger.info("="*80 + "\n")
    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")



def main():
    logger.info("="*80)
    logger.info("YOLOV11 SEGMENTATION INFERENCE ON TEST SET")
    logger.info("="*80)

    config = InferenceConfig()
    config.__post_init__()

    logger.info(f"Loading model from: {config.MODEL_PATH}")
    model = YOLO(str(config.MODEL_PATH))

    run_test_val(config, model)

    run_inference_with_visualization(config, model)

    logger.info("="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
