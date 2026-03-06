"""
SAM (Segment Anything Model) Refinement for YOLO Predictions
=============================================================
This script:
- Loads YOLO predictions (bounding boxes) from external patch inference
- Uses SAM to generate high-quality segmentation masks from bounding boxes
- Transforms SAM mask coordinates to:
  1. Resized image space (1024×768)
  2. Original patch space
  3. WSI coordinate space
- Compares YOLO masks vs SAM masks
- Saves all results with coordinate mappings

Usage:
    python 6_sam_refinement_external.py

Requirements:
    - ultralytics (with SAM support)
    - torch with CUDA support
    - opencv-python
    - numpy
    - pandas
    - matplotlib
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import SAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sam_refinement")


class SAMRefinementConfig:
    """Configuration for SAM refinement"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    
    SAM_MODEL = "sam_l.pt"
    
    YOLO_RESULTS_DIR = BASE_DIR / "results/external_inference_A2"
    YOLO_RESULTS_JSON = YOLO_RESULTS_DIR / "all_patches_results.json"
    
    PATCHES_DIR = BASE_DIR / "dataset_yolo2/test_external/large_intestine/final_patches_A2_nm"
    
    OUTPUT_DIR = BASE_DIR / "results/sam_refined_A2"
    VIS_DIR = OUTPUT_DIR / "visualizations"
    COORDS_DIR = OUTPUT_DIR / "coordinates"
    COMPARISON_DIR = OUTPUT_DIR / "yolo_vs_sam_comparison"
    
    TARGET_WIDTH = 1024
    TARGET_HEIGHT = 768
    
    DEVICE = 0
    
    COLORS = {
        'crypt': (255, 0, 0),
        'gland': (0, 255, 0),
        'yolo': (255, 255, 0),
        'sam': (0, 255, 255),
        'text': (255, 255, 255),
        'bbox': 2,
        'mask_alpha': 0.4,
    }
    
    CLASS_NAMES = ['crypt', 'gland']
    
    WSI_NAME = "A1_WSI"
    
    def __post_init__(self):
        """Validate configuration and create directories"""
        if not self.YOLO_RESULTS_JSON.exists():
            raise FileNotFoundError(
                f"YOLO results not found: {self.YOLO_RESULTS_JSON}\n"
                "Please run 5_inference_external_patches.py first!"
            )
        
        if not self.PATCHES_DIR.exists():
            raise FileNotFoundError(f"Patches directory not found: {self.PATCHES_DIR}")
        
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VIS_DIR.mkdir(parents=True, exist_ok=True)
        self.COORDS_DIR.mkdir(parents=True, exist_ok=True)
        self.COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; using CPU (will be slow)")
            self.DEVICE = 'cpu'



class CoordinateTransformer:
    """Handles coordinate transformations between different spaces"""
    
    def __init__(self, original_width: int, original_height: int, 
                 resized_width: int, resized_height: int, 
                 patch_xmin: int, patch_ymin: int):
        self.original_width = original_width
        self.original_height = original_height
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.patch_xmin = patch_xmin
        self.patch_ymin = patch_ymin
        
        self.scale_x = original_width / resized_width
        self.scale_y = original_height / resized_height
    
    def resized_to_original_patch(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates from resized image to original patch"""
        coords_transformed = coords.copy()
        coords_transformed[:, 0::2] *= self.scale_x
        coords_transformed[:, 1::2] *= self.scale_y
        return coords_transformed
    
    def original_patch_to_wsi(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates from original patch to WSI"""
        coords_transformed = coords.copy()
        coords_transformed[:, 0::2] += self.patch_xmin
        coords_transformed[:, 1::2] += self.patch_ymin
        return coords_transformed
    
    def polygon_resized_to_original_patch(self, polygon: np.ndarray) -> np.ndarray:
        """Transform polygon points from resized to original patch space"""
        polygon_transformed = polygon.astype(np.float64)
        polygon_transformed[:, 0] *= self.scale_x
        polygon_transformed[:, 1] *= self.scale_y
        return polygon_transformed
    
    def polygon_original_patch_to_wsi(self, polygon: np.ndarray) -> np.ndarray:
        """Transform polygon points from original patch to WSI space"""
        polygon_transformed = polygon.astype(np.float64)
        polygon_transformed[:, 0] += self.patch_xmin
        polygon_transformed[:, 1] += self.patch_ymin
        return polygon_transformed



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


def generate_sam_masks_from_boxes(
    sam_model: SAM,
    image: np.ndarray,
    boxes: np.ndarray,
    config: SAMRefinementConfig
) -> List[np.ndarray]:
    """
    Generate SAM masks from bounding boxes.
    
    Args:
        sam_model: Loaded SAM model
        image: Image (H, W, 3) in BGR
        boxes: Bounding boxes (N, 4) in xyxy format
        config: Configuration object
    
    Returns:
        List of masks (N, H, W)
    """
    if len(boxes) == 0:
        return []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = sam_model(image_rgb, bboxes=boxes, verbose=False)
    
    if results and len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        return masks
    else:
        logger.warning("SAM did not return any masks")
        return []



def create_yolo_vs_sam_comparison(
    image: np.ndarray,
    yolo_mask: np.ndarray,
    sam_mask: np.ndarray,
    bbox: np.ndarray,
    class_name: str,
    confidence: float,
    save_path: Path,
    config: SAMRefinementConfig
):
    """
    Create side-by-side comparison of YOLO mask vs SAM mask.
    
    Args:
        image: Original image (H, W, 3) BGR
        yolo_mask: YOLO mask (H, W)
        sam_mask: SAM mask (H, W)
        bbox: Bounding box (4,) xyxy
        class_name: Class name
        confidence: Detection confidence
        save_path: Path to save comparison
        config: Configuration
    """
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


def create_patch_summary_visualization(
    original_img: np.ndarray,
    resized_img: np.ndarray,
    yolo_annotated: np.ndarray,
    sam_annotated: np.ndarray,
    patch_name: str,
    save_path: Path
):
    """Create 4-panel summary for a patch"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'SAM Refinement Results: {patch_name}', fontsize=16, fontweight='bold')
    
    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    yolo_rgb = cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB)
    sam_rgb = cv2.cvtColor(sam_annotated, cv2.COLOR_BGR2RGB)
    
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title(f'Original Patch\n{original_img.shape[1]}×{original_img.shape[0]}', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(resized_rgb)
    axes[0, 1].set_title(f'Resized for Inference\n1024×768', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(yolo_rgb)
    axes[1, 0].set_title('YOLO Predictions\n(Original Masks)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(sam_rgb)
    axes[1, 1].set_title('SAM Refined Masks\n(Higher Quality)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def overlay_masks_on_image(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    config: SAMRefinementConfig,
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



def refine_with_sam(config: SAMRefinementConfig):
    """
    Main function to refine YOLO predictions with SAM.
    """
    logger.info("="*80)
    logger.info("SAM REFINEMENT OF YOLO PREDICTIONS")
    logger.info("="*80)
    
    logger.info(f"Loading YOLO results from: {config.YOLO_RESULTS_JSON}")
    with open(config.YOLO_RESULTS_JSON, 'r') as f:
        yolo_results = json.load(f)
    
    logger.info(f"Found {yolo_results['num_patches']} patches with {yolo_results['total_detections']} detections")
    
    logger.info(f"Loading SAM model: {config.SAM_MODEL}")
    sam_model = SAM(config.SAM_MODEL)
    sam_model.to(config.DEVICE)
    logger.info("✓ SAM model loaded successfully")
    
    all_refined_results = []
    
    for patch_idx, patch_result in enumerate(yolo_results['patches']):
        patch_name = patch_result['patch_name']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {patch_idx + 1}/{yolo_results['num_patches']}: {patch_name}")
        logger.info(f"{'='*80}")
        
        if patch_result['num_detections'] == 0:
            logger.info("No detections to refine, skipping")
            all_refined_results.append(patch_result)
            continue
        
        patch_path = config.PATCHES_DIR / patch_name
        if not patch_path.exists():
            logger.warning(f"Patch file not found: {patch_path}")
            continue
        
        original_img = cv2.imread(str(patch_path))
        
        pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        resized_pil = pil_img.resize((config.TARGET_WIDTH, config.TARGET_HEIGHT), Image.LANCZOS)
        resized_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
        
        patch_info = patch_result['patch_info']
        transformer = CoordinateTransformer(
            patch_info['original_width'],
            patch_info['original_height'],
            config.TARGET_WIDTH,
            config.TARGET_HEIGHT,
            patch_info['wsi_xmin'],
            patch_info['wsi_ymin']
        )
        
        boxes_resized = []
        classes = []
        confidences = []
        
        for det in patch_result['detections']:
            bbox = det['bbox_resized']
            boxes_resized.append([bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
            classes.append(config.CLASS_NAMES.index(det['class']))
            confidences.append(det['confidence'])
        
        boxes_resized = np.array(boxes_resized)
        classes = np.array(classes)
        confidences = np.array(confidences)
        
        logger.info(f"Generating SAM masks for {len(boxes_resized)} detections...")
        
        sam_masks = generate_sam_masks_from_boxes(sam_model, resized_img, boxes_resized, config)
        
        if len(sam_masks) == 0:
            logger.warning("SAM did not generate any masks")
            all_refined_results.append(patch_result)
            continue
        
        logger.info(f"✓ Generated {len(sam_masks)} SAM masks")
        
        refined_detections = []
        
        for det_idx, det in enumerate(patch_result['detections']):
            if det_idx >= len(sam_masks):
                logger.warning(f"No SAM mask for detection {det_idx}")
                refined_detections.append(det)
                continue
            
            sam_mask = sam_masks[det_idx]
            
            sam_polygons_resized = mask_to_polygon(sam_mask)
            
            refined_det = det.copy()
            refined_det['sam_segmentation_resized'] = []
            refined_det['sam_segmentation_original_patch'] = []
            refined_det['sam_segmentation_wsi'] = []
            
            for poly_resized in sam_polygons_resized:
                poly_patch = transformer.polygon_resized_to_original_patch(poly_resized)
                poly_wsi = transformer.polygon_original_patch_to_wsi(poly_patch)
                
                refined_det['sam_segmentation_resized'].append(poly_resized.tolist())
                refined_det['sam_segmentation_original_patch'].append(poly_patch.tolist())
                refined_det['sam_segmentation_wsi'].append(poly_wsi.tolist())
            
            if 'segmentation_resized' in det:
                refined_det['yolo_segmentation_resized'] = det['segmentation_resized']
                refined_det['yolo_segmentation_original_patch'] = det['segmentation_original_patch']
                refined_det['yolo_segmentation_wsi'] = det['segmentation_wsi']
                del refined_det['segmentation_resized']
                del refined_det['segmentation_original_patch']
                del refined_det['segmentation_wsi']
            
            refined_detections.append(refined_det)
            
            yolo_mask = None
            if 'yolo_segmentation_resized' in refined_det and refined_det['yolo_segmentation_resized']:
                yolo_mask = np.zeros((config.TARGET_HEIGHT, config.TARGET_WIDTH), dtype=np.uint8)
                for poly in refined_det['yolo_segmentation_resized']:
                    poly_np = np.array(poly, dtype=np.int32)
                    cv2.fillPoly(yolo_mask, [poly_np], 255)
            
            comparison_path = config.COMPARISON_DIR / f"{Path(patch_name).stem}_det{det_idx}_{det['class']}.png"
            create_yolo_vs_sam_comparison(
                resized_img, yolo_mask, sam_mask, boxes_resized[det_idx],
                det['class'], det['confidence'], comparison_path, config
            )
        
        refined_patch_result = {
            'patch_name': patch_name,
            'patch_info': patch_info,
            'num_detections': len(refined_detections),
            'detections': refined_detections,
            'refinement_method': 'SAM'
        }
        
        all_refined_results.append(refined_patch_result)
        
        json_path = config.COORDS_DIR / f"{Path(patch_name).stem}_sam_refined.json"
        with open(json_path, 'w') as f:
            json.dump(refined_patch_result, f, indent=2)
        logger.info(f"Saved: {json_path}")
        
        yolo_annotated = overlay_masks_on_image(
            resized_img, boxes_resized, 
            [np.zeros((config.TARGET_HEIGHT, config.TARGET_WIDTH))]*len(boxes_resized),
            classes, confidences, config, "[YOLO]"
        )
        
        sam_annotated = overlay_masks_on_image(
            resized_img, boxes_resized, sam_masks,
            classes, confidences, config, "[SAM]"
        )
        
        summary_path = config.VIS_DIR / f"{Path(patch_name).stem}_sam_summary.png"
        create_patch_summary_visualization(
            original_img, resized_img, yolo_annotated, sam_annotated,
            patch_name, summary_path
        )
        logger.info(f"Saved: {summary_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("SAVING CONSOLIDATED SAM-REFINED RESULTS")
    logger.info(f"{'='*80}")
    
    consolidated_json = config.OUTPUT_DIR / "all_patches_sam_refined.json"
    with open(consolidated_json, 'w') as f:
        json.dump({
            'wsi_info': yolo_results['wsi_info'],
            'num_patches': len(all_refined_results),
            'total_detections': sum(r['num_detections'] for r in all_refined_results),
            'timestamp': datetime.now().isoformat(),
            'refinement_method': 'SAM (Segment Anything Model)',
            'patches': all_refined_results
        }, f, indent=2)
    logger.info(f"Saved: {consolidated_json}")
    
    summary_path = config.OUTPUT_DIR / "sam_refinement_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SAM REFINEMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total patches: {len(all_refined_results)}\n")
        f.write(f"Patches with detections: {sum(1 for r in all_refined_results if r['num_detections'] > 0)}\n")
        f.write(f"Total detections refined: {sum(r['num_detections'] for r in all_refined_results)}\n")
        f.write(f"\nRefinement method: SAM (Segment Anything Model)\n")
        f.write(f"Model: {config.SAM_MODEL}\n")
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"\n{'='*80}")
    logger.info("SAM REFINEMENT COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Visualizations: {config.VIS_DIR}")
    logger.info(f"YOLO vs SAM comparisons: {config.COMPARISON_DIR}")
    logger.info(f"Coordinates: {config.COORDS_DIR}")
    logger.info(f"{'='*80}\n")
    
    return all_refined_results



def main():
    config = SAMRefinementConfig()
    config.__post_init__()
    
    try:
        results = refine_with_sam(config)
        logger.info("✓ SAM refinement completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"SAM refinement failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
