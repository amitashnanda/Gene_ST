"""
SAM Refinement for YOLO Test Set Predictions
=============================================
This script refines YOLO predictions on the test set using SAM.
Since Script 4 only saved visualizations, this script will:
1. Run YOLO inference on test set
2. Save predictions in structured JSON format
3. Generate SAM refinements for each detection
4. Create YOLO vs SAM comparison visualizations

Usage:
    python 7_sam_refinement_test_set.py

Requirements:
    - ultralytics (with SAM support)
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO, SAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sam_test_refinement")


class SAMTestRefinementConfig:
    """Configuration for SAM refinement on test set"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    
    YOLO_MODEL_PATH = BASE_DIR / "runs/yolo11m_seg_dinov3_20251208_093550/weights/last.pt"
    SAM_MODEL = "sam_l.pt"
    
    DATA_YAML = BASE_DIR / "dataset_yolo2/data.yaml"
    
    OUTPUT_DIR = BASE_DIR / "results/sam_refined_test_set"
    VIS_DIR = OUTPUT_DIR / "visualizations"
    COMPARISON_DIR = OUTPUT_DIR / "yolo_vs_sam_comparison"
    COORDS_DIR = OUTPUT_DIR / "coordinates"
    
    IMAGE_SIZE = (768, 1024)
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    DEVICE = 0
    
    COLORS = {
        'crypt': (255, 0, 0),
        'gland': (0, 255, 0),
        'text': (255, 255, 255),
        'bbox': 2,
        'mask_alpha': 0.4,
    }
    
    CLASS_NAMES = ['crypt', 'gland']
    
    def __post_init__(self):
        if not self.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.YOLO_MODEL_PATH}")
        if not self.DATA_YAML.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.DATA_YAML}")
        
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VIS_DIR.mkdir(parents=True, exist_ok=True)
        self.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        self.COORDS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; using CPU")
            self.DEVICE = 'cpu'



def mask_to_polygon(mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """Convert binary mask to polygon coordinates"""
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        polygon = contour.reshape(-1, 2)
        polygons.append(polygon)
    
    return polygons


def create_comparison_visualization(
    image: np.ndarray,
    yolo_mask: np.ndarray,
    sam_mask: np.ndarray,
    bbox: np.ndarray,
    class_name: str,
    confidence: float,
    save_path: Path,
    config: SAMTestRefinementConfig
):
    """Create comprehensive YOLO vs SAM comparison"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'{class_name} (conf: {confidence:.2f}) - YOLO vs SAM Comparison', 
                 fontsize=14, fontweight='bold')
    
    axes[0].imshow(image_rgb)
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                         edgecolor='red', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title('Original + BBox', fontsize=12)
    axes[0].axis('off')
    
    yolo_overlay = image_rgb.copy()
    if yolo_mask is not None and yolo_mask.shape[:2] == image_rgb.shape[:2]:
        mask_bool = yolo_mask > 0.5
        yolo_overlay[mask_bool] = yolo_overlay[mask_bool] * 0.5 + np.array([255, 255, 0]) * 0.5
    axes[1].imshow(yolo_overlay.astype(np.uint8))
    axes[1].set_title('YOLO Mask', fontsize=12)
    axes[1].axis('off')
    
    sam_overlay = image_rgb.copy()
    if sam_mask is not None and sam_mask.shape[:2] == image_rgb.shape[:2]:
        mask_bool = sam_mask > 0.5
        sam_overlay[mask_bool] = sam_overlay[mask_bool] * 0.5 + np.array([0, 255, 255]) * 0.5
    axes[2].imshow(sam_overlay.astype(np.uint8))
    axes[2].set_title('SAM Mask (Refined)', fontsize=12)
    axes[2].axis('off')
    
    comparison = image_rgb.copy()
    if yolo_mask is not None and yolo_mask.shape[:2] == image_rgb.shape[:2]:
        yolo_bool = yolo_mask > 0.5
        comparison[yolo_bool] = comparison[yolo_bool] * 0.7 + np.array([255, 255, 0]) * 0.3
    if sam_mask is not None and sam_mask.shape[:2] == image_rgb.shape[:2]:
        sam_bool = sam_mask > 0.5
        comparison[sam_bool] = comparison[sam_bool] * 0.7 + np.array([0, 255, 255]) * 0.3
    axes[3].imshow(comparison.astype(np.uint8))
    axes[3].set_title('Overlay (Yellow=YOLO, Cyan=SAM)', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(
    image: np.ndarray,
    yolo_annotated: np.ndarray,
    sam_annotated: np.ndarray,
    image_name: str,
    save_path: Path
):
    """Create 3-panel summary for an image"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'SAM Refinement Results: {image_name}', fontsize=16, fontweight='bold')
    
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_rgb = cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB)
    sam_rgb = cv2.cvtColor(sam_annotated, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(yolo_rgb)
    axes[1].set_title('YOLO Predictions', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(sam_rgb)
    axes[2].set_title('SAM Refined', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def overlay_masks_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    config: SAMTestRefinementConfig,
    mask_label: str = ""
) -> np.ndarray:
    """Overlay masks and boxes on image"""
    annotated = image.copy()
    h, w = image.shape[:2]
    
    if boxes is None or len(boxes) == 0:
        return annotated
    
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    
    for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
        cls_idx = int(cls)
        class_name = config.CLASS_NAMES[cls_idx]
        color = config.COLORS[class_name]
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, config.COLORS['bbox'])
        
        label = f"{class_name} {conf:.2f} {mask_label}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1 - 10, label_size[1] + 10)
        
        cv2.rectangle(annotated, (x1, label_y - label_size[1] - 5),
                     (x1 + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(annotated, label, (x1 + 2, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLORS['text'], 2)
        
        if masks is not None and idx < len(masks):
            mask = masks[idx]
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                                 interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0.5
            mask_overlay[mask_bool] = color
    
    annotated = cv2.addWeighted(annotated, 1.0, mask_overlay, 
                               config.COLORS['mask_alpha'], 0)
    
    return annotated



def main():
    config = SAMTestRefinementConfig()
    config.__post_init__()
    
    logger.info("="*80)
    logger.info("SAM REFINEMENT FOR TEST SET")
    logger.info("="*80)
    logger.info("Note: Script 4 only saved visualizations, so we'll run YOLO inference")
    logger.info("      and save structured predictions for future use.")
    logger.info("="*80)
    
    with open(config.DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    
    test_path = Path(data['test'])
    if not test_path.is_absolute():
        test_path = (config.DATA_YAML.parent / test_path).resolve()
    
    logger.info(f"Test images directory: {test_path}")
    
    image_files = sorted(list(test_path.glob("*.png")) + list(test_path.glob("*.jpg")))
    logger.info(f"Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        logger.error("No test images found!")
        return 1
    
    logger.info(f"\nLoading YOLO model: {config.YOLO_MODEL_PATH}")
    yolo_model = YOLO(str(config.YOLO_MODEL_PATH))
    
    logger.info(f"Loading SAM model: {config.SAM_MODEL}")
    sam_model = SAM(config.SAM_MODEL)
    sam_model.to(config.DEVICE)
    
    logger.info("✓ Models loaded successfully")
    
    all_results = []
    results_summary = []
    
    for img_idx, img_path in enumerate(image_files):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {img_idx + 1}/{len(image_files)}: {img_path.name}")
        logger.info(f"{'='*80}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to load {img_path}")
            continue
        
        img_h, img_w = image.shape[:2]
        logger.info(f"Image size: {img_w}×{img_h}")
        
        logger.info("Running YOLO inference...")
        yolo_results = yolo_model.predict(
            image,
            imgsz=config.IMAGE_SIZE,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            save=False,
            verbose=False
        )
        
        result = yolo_results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            logger.info("  No detections found")
            
            image_result = {
                'image_name': img_path.name,
                'image_size': {'width': img_w, 'height': img_h},
                'num_detections': 0,
                'detections': []
            }
            all_results.append(image_result)
            continue
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        yolo_masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        
        logger.info(f"  ✓ Found {len(boxes)} YOLO detections")
        
        logger.info(f"  Generating SAM masks for {len(boxes)} detections...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_results = sam_model(image_rgb, bboxes=boxes, verbose=False)
        
        if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
            sam_masks = sam_results[0].masks.data.cpu().numpy()
            logger.info(f"  ✓ Generated {len(sam_masks)} SAM masks")
        else:
            logger.warning("  SAM did not generate masks")
            sam_masks = None
        
        detections = []
        
        for det_idx in range(len(boxes)):
            bbox = boxes[det_idx]
            cls = int(classes[det_idx])
            conf = float(confidences[det_idx])
            class_name = config.CLASS_NAMES[cls]
            
            yolo_mask = yolo_masks[det_idx] if yolo_masks is not None else None
            sam_mask = sam_masks[det_idx] if sam_masks is not None else None
            
            yolo_polygons = []
            sam_polygons = []
            
            if yolo_mask is not None:
                if yolo_mask.shape != (img_h, img_w):
                    yolo_mask_resized = cv2.resize(yolo_mask, (img_w, img_h), 
                                                   interpolation=cv2.INTER_NEAREST)
                else:
                    yolo_mask_resized = yolo_mask
                yolo_polygons = mask_to_polygon(yolo_mask_resized)
            
            if sam_mask is not None:
                if sam_mask.shape != (img_h, img_w):
                    sam_mask_resized = cv2.resize(sam_mask, (img_w, img_h), 
                                                  interpolation=cv2.INTER_NEAREST)
                else:
                    sam_mask_resized = sam_mask
                sam_polygons = mask_to_polygon(sam_mask_resized)
            
            detection_data = {
                'detection_id': det_idx,
                'class': class_name,
                'confidence': conf,
                'bbox': {
                    'xmin': float(bbox[0]),
                    'ymin': float(bbox[1]),
                    'xmax': float(bbox[2]),
                    'ymax': float(bbox[3]),
                    'width': float(bbox[2] - bbox[0]),
                    'height': float(bbox[3] - bbox[1])
                },
                'yolo_segmentation': [poly.tolist() for poly in yolo_polygons],
                'sam_segmentation': [poly.tolist() for poly in sam_polygons],
                'has_yolo_mask': yolo_mask is not None,
                'has_sam_mask': sam_mask is not None
            }
            detections.append(detection_data)
            
            if yolo_mask is not None or sam_mask is not None:
                comparison_path = config.COMPARISON_DIR / f"{img_path.stem}_det{det_idx}_{class_name}.png"
                create_comparison_visualization(
                    image, 
                    yolo_mask_resized if yolo_mask is not None else None,
                    sam_mask_resized if sam_mask is not None else None,
                    bbox, class_name, conf, comparison_path, config
                )
                logger.info(f"  Saved: {comparison_path.name}")
        
        image_result = {
            'image_name': img_path.name,
            'image_size': {'width': img_w, 'height': img_h},
            'num_detections': len(detections),
            'detections': detections
        }
        all_results.append(image_result)
        
        json_path = config.COORDS_DIR / f"{img_path.stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(image_result, f, indent=2)
        
        yolo_annotated = overlay_masks_on_image(
            image, boxes, yolo_masks, classes, confidences, config, "[YOLO]"
        )
        
        sam_annotated = overlay_masks_on_image(
            image, boxes, sam_masks, classes, confidences, config, "[SAM]"
        )
        
        summary_path = config.VIS_DIR / f"{img_path.stem}_summary.png"
        create_summary_visualization(
            image, yolo_annotated, sam_annotated, img_path.name, summary_path
        )
        logger.info(f"  Saved: {summary_path.name}")
        
        results_summary.append({
            'image': img_path.name,
            'yolo_detections': len(boxes),
            'sam_masks_generated': len(sam_masks) if sam_masks is not None else 0,
            'crypts': sum(1 for d in detections if d['class'] == 'crypt'),
            'glands': sum(1 for d in detections if d['class'] == 'gland')
        })
    
    logger.info(f"\n{'='*80}")
    logger.info("SAVING CONSOLIDATED RESULTS")
    logger.info(f"{'='*80}")
    
    consolidated_json = config.OUTPUT_DIR / "all_test_images_sam_refined.json"
    with open(consolidated_json, 'w') as f:
        json.dump({
            'num_images': len(all_results),
            'total_detections': sum(r['num_detections'] for r in all_results),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'yolo_model': str(config.YOLO_MODEL_PATH),
                'sam_model': config.SAM_MODEL
            },
            'images': all_results
        }, f, indent=2)
    logger.info(f"Saved: {consolidated_json}")
    
    df_summary = pd.DataFrame(results_summary)
    summary_csv = config.OUTPUT_DIR / "refinement_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    logger.info(f"Saved: {summary_csv}")
    
    summary_txt = config.OUTPUT_DIR / "sam_refinement_summary.txt"
    with open(summary_txt, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAM REFINEMENT SUMMARY - TEST SET\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total images: {len(all_results)}\n")
        f.write(f"Images with detections: {sum(1 for r in all_results if r['num_detections'] > 0)}\n")
        f.write(f"Total detections: {sum(r['num_detections'] for r in all_results)}\n")
        f.write(f"\nDetection breakdown:\n")
        f.write(f"  Crypts: {df_summary['crypts'].sum()}\n")
        f.write(f"  Glands: {df_summary['glands'].sum()}\n")
        f.write(f"\nModels:\n")
        f.write(f"  YOLO: {config.YOLO_MODEL_PATH.name}\n")
        f.write(f"  SAM: {config.SAM_MODEL}\n")
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"\n{'='*80}")
    logger.info("SAM REFINEMENT COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Visualizations: {config.VIS_DIR}")
    logger.info(f"YOLO vs SAM comparisons: {config.COMPARISON_DIR}")
    logger.info(f"Coordinates (JSON): {config.COORDS_DIR}")
    logger.info(f"Summary CSV: {summary_csv}")
    logger.info(f"Consolidated JSON: {consolidated_json}")
    logger.info(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"SAM refinement failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
