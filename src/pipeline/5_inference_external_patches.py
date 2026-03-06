"""
YOLOv11 Inference on External WSI Patches with Coordinate Transformation
========================================================================
This script:
- Loads patches extracted from a large WSI image
- Resizes patches to 1024×1024 using Lanczos interpolation (lossless)
- Runs YOLOv11 inference on resized patches
- Saves visualizations with bounding boxes and segmentation masks
- Transforms coordinates back to:
  1. Resized image coordinates (1024×1024)
  2. Original patch coordinates
  3. Original WSI coordinates
- Saves all results in structured format

Usage:
    python 5_inference_external_patches.py

Requirements:
    - ultralytics
    - torch with CUDA support
    - opencv-python
    - numpy
    - pandas
    - matplotlib
    - Pillow
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
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("external_inference")


class ExternalInferenceConfig:
    """Configuration for external patch inference"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    
    MODEL_PATH = BASE_DIR / "runs/yolo11m_seg_dinov3_20251208_093550/weights/last.pt"
    
    PATCHES_DIR = BASE_DIR / "dataset_yolo2/test_external/large_intestine/final_patches_A2_nm"
    COORDS_FILE = PATCHES_DIR / "patch_coords.txt"
    OUTPUT_DIR = BASE_DIR / "results/external_inference_A2"
    VIS_DIR = OUTPUT_DIR / "visualizations"
    COORDS_DIR = OUTPUT_DIR / "coordinates"
    
    TARGET_WIDTH = 1024
    TARGET_HEIGHT = 768
    INTERPOLATION = Image.LANCZOS
    
    CONF_THRESHOLD = 0.3
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
    
    WSI_WIDTH = 6997
    WSI_HEIGHT = 7084
    WSI_NAME = "A1_WSI"
    
    def __post_init__(self):
        """Validate configuration and create directories"""
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")
        
        if not self.PATCHES_DIR.exists():
            raise FileNotFoundError(f"Patches directory not found: {self.PATCHES_DIR}")
        
        if not self.COORDS_FILE.exists():
            raise FileNotFoundError(f"Coordinates file not found: {self.COORDS_FILE}")
        
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.VIS_DIR.mkdir(parents=True, exist_ok=True)
        self.COORDS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; using CPU")
            self.DEVICE = 'cpu'



class CoordinateTransformer:
    """Handles coordinate transformations between different spaces"""
    
    def __init__(self, original_width: int, original_height: int, 
                 resized_width: int, resized_height: int, patch_xmin: int, patch_ymin: int):
        """
        Initialize coordinate transformer.
        
        Args:
            original_width: Width of original patch
            original_height: Height of original patch
            resized_width: Width of resized image (1024)
            resized_height: Height of resized image (768)
            patch_xmin: X coordinate of patch in WSI
            patch_ymin: Y coordinate of patch in WSI
        """
        self.original_width = original_width
        self.original_height = original_height
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.patch_xmin = patch_xmin
        self.patch_ymin = patch_ymin
        
        self.scale_x = original_width / resized_width
        self.scale_y = original_height / resized_height
    
    def resized_to_original_patch(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from resized image to original patch.
        
        Args:
            coords: Array of coordinates (N, 2) or (N, 4) in resized space
        
        Returns:
            Transformed coordinates in original patch space
        """
        coords_transformed = coords.copy()
        coords_transformed[:, 0::2] *= self.scale_x
        coords_transformed[:, 1::2] *= self.scale_y
        return coords_transformed
    
    def original_patch_to_wsi(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from original patch to WSI.
        
        Args:
            coords: Array of coordinates (N, 2) or (N, 4) in patch space
        
        Returns:
            Transformed coordinates in WSI space
        """
        coords_transformed = coords.copy()
        coords_transformed[:, 0::2] += self.patch_xmin
        coords_transformed[:, 1::2] += self.patch_ymin
        return coords_transformed
    
    def resized_to_wsi(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from resized image directly to WSI.
        
        Args:
            coords: Array of coordinates in resized space
        
        Returns:
            Transformed coordinates in WSI space
        """
        patch_coords = self.resized_to_original_patch(coords)
        return self.original_patch_to_wsi(patch_coords)
    
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



def load_patch_coordinates(coords_file: Path) -> pd.DataFrame:
    """
    Load patch coordinates from text file.
    
    Args:
        coords_file: Path to patch_coords.txt
    
    Returns:
        DataFrame with patch information
    """
    df = pd.read_csv(coords_file, sep=' ')
    logger.info(f"Loaded coordinates for {len(df)} patches")
    return df



def overlay_detections_on_image(
    image: np.ndarray,
    boxes: Optional[np.ndarray],
    masks: Optional[np.ndarray],
    classes: Optional[np.ndarray],
    confidences: Optional[np.ndarray],
    config: ExternalInferenceConfig
) -> np.ndarray:
    """
    Overlay detections (boxes and masks) on image.
    
    Args:
        image: Original image (H, W, 3) BGR
        boxes: Bounding boxes (N, 4) xyxy format
        masks: Segmentation masks (N, H, W)
        classes: Class indices (N,)
        confidences: Confidence scores (N,)
        config: Configuration object
    
    Returns:
        Annotated image in BGR format
    """
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
        
        label = f"{class_name} {conf:.2f}"
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


def save_visualization_comparison(
    original: np.ndarray,
    resized: np.ndarray,
    annotated: np.ndarray,
    patch_name: str,
    save_path: Path
):
    """
    Create 3-panel comparison: original patch, resized, and annotated.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f'Inference Results: {patch_name}', fontsize=16, fontweight='bold')
    
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(original_rgb)
    axes[0].set_title(f'Original Patch\n{original.shape[1]}×{original.shape[0]}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(resized_rgb)
    axes[1].set_title(f'Resized for Inference\n1024×768', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(annotated_rgb)
    axes[2].set_title('Predictions\n(Boxes + Masks)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



def mask_to_polygon(mask: np.ndarray, min_area: int = 50) -> List[np.ndarray]:
    """
    Convert binary mask to polygon coordinates.
    
    Args:
        mask: Binary mask (H, W)
        min_area: Minimum contour area to keep
    
    Returns:
        List of polygon arrays (N, 2) where N is number of points
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        
        polygon = contour.reshape(-1, 2)
        polygons.append(polygon)
    
    return polygons



def run_inference_on_external_patches(config: ExternalInferenceConfig):
    """
    Main function to run inference on external WSI patches.
    """
    logger.info("="*80)
    logger.info("YOLOV11 INFERENCE ON EXTERNAL WSI PATCHES")
    logger.info("="*80)
    
    logger.info(f"Loading model: {config.MODEL_PATH}")
    model = YOLO(str(config.MODEL_PATH))
    
    patch_coords = load_patch_coordinates(config.COORDS_FILE)
    
    patch_files = sorted(config.PATCHES_DIR.glob("patch_*.png"))
    logger.info(f"Found {len(patch_files)} patches to process")
    
    all_results = []
    
    for patch_idx, patch_file in enumerate(patch_files):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {patch_idx + 1}/{len(patch_files)}: {patch_file.name}")
        logger.info(f"{'='*80}")
        
        patch_info = patch_coords[patch_coords['patch_name'] == patch_file.name]
        if patch_info.empty:
            logger.warning(f"No coordinate info for {patch_file.name}, skipping")
            continue
        
        patch_info = patch_info.iloc[0]
        original_width = int(patch_info['width'])
        original_height = int(patch_info['height'])
        patch_xmin = int(patch_info['xmin'])
        patch_ymin = int(patch_info['ymin'])
        
        logger.info(f"Original size: {original_width}×{original_height}")
        logger.info(f"WSI position: ({patch_xmin}, {patch_ymin})")
        
        original_img = cv2.imread(str(patch_file))
        if original_img is None:
            logger.error(f"Failed to load {patch_file}")
            continue
        
        pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        resized_pil = pil_img.resize((config.TARGET_WIDTH, config.TARGET_HEIGHT), 
                                     config.INTERPOLATION)
        resized_img = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Resized to: {config.TARGET_WIDTH}×{config.TARGET_HEIGHT}")
        
        transformer = CoordinateTransformer(
            original_width, original_height, config.TARGET_WIDTH, config.TARGET_HEIGHT,
            patch_xmin, patch_ymin
        )
        
        logger.info("Running inference...")
        results = model.predict(
            resized_img,
            imgsz=(config.TARGET_HEIGHT, config.TARGET_WIDTH),
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            save=False,
            verbose=False,
        )
        
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_resized = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            if result.masks is not None:
                masks_resized = result.masks.data.cpu().numpy()
            else:
                masks_resized = None
            
            logger.info(f"Found {len(boxes_resized)} detections")
            
            boxes_patch = transformer.resized_to_original_patch(boxes_resized)
            boxes_wsi = transformer.original_patch_to_wsi(boxes_patch)
            
            detections = []
            for det_idx, (box_r, box_p, box_w, cls, conf) in enumerate(
                zip(boxes_resized, boxes_patch, boxes_wsi, classes, confidences)
            ):
                detection = {
                    'detection_id': det_idx,
                    'class': config.CLASS_NAMES[int(cls)],
                    'confidence': float(conf),
                    
                    'bbox_resized': {
                        'xmin': float(box_r[0]), 'ymin': float(box_r[1]),
                        'xmax': float(box_r[2]), 'ymax': float(box_r[3]),
                        'width': float(box_r[2] - box_r[0]),
                        'height': float(box_r[3] - box_r[1])
                    },
                    'bbox_original_patch': {
                        'xmin': float(box_p[0]), 'ymin': float(box_p[1]),
                        'xmax': float(box_p[2]), 'ymax': float(box_p[3]),
                        'width': float(box_p[2] - box_p[0]),
                        'height': float(box_p[3] - box_p[1])
                    },
                    'bbox_wsi': {
                        'xmin': float(box_w[0]), 'ymin': float(box_w[1]),
                        'xmax': float(box_w[2]), 'ymax': float(box_w[3]),
                        'width': float(box_w[2] - box_w[0]),
                        'height': float(box_w[3] - box_w[1])
                    },
                }
                
                if masks_resized is not None and det_idx < len(masks_resized):
                    mask = masks_resized[det_idx]
                    
                    polygons_resized = mask_to_polygon(mask)
                    
                    detection['segmentation_resized'] = []
                    detection['segmentation_original_patch'] = []
                    detection['segmentation_wsi'] = []
                    
                    for poly_resized in polygons_resized:
                        poly_patch = transformer.polygon_resized_to_original_patch(poly_resized)
                        poly_wsi = transformer.polygon_original_patch_to_wsi(poly_patch)
                        
                        detection['segmentation_resized'].append(poly_resized.tolist())
                        detection['segmentation_original_patch'].append(poly_patch.tolist())
                        detection['segmentation_wsi'].append(poly_wsi.tolist())
                
                detections.append(detection)
            
            patch_result = {
                'patch_name': patch_file.name,
                'patch_info': {
                    'original_width': original_width,
                    'original_height': original_height,
                    'wsi_xmin': patch_xmin,
                    'wsi_ymin': patch_ymin,
                    'wsi_xmax': int(patch_info['xmax']),
                    'wsi_ymax': int(patch_info['ymax']),
                },
                'num_detections': len(detections),
                'detections': detections
            }
            
            all_results.append(patch_result)
            
            json_path = config.COORDS_DIR / f"{patch_file.stem}_results.json"
            with open(json_path, 'w') as f:
                json.dump(patch_result, f, indent=2)
            logger.info(f"Saved results: {json_path}")
            
            csv_path = config.COORDS_DIR / f"{patch_file.stem}_detections.csv"
            detections_flat = []
            for det in detections:
                det_flat = {
                    'detection_id': det['detection_id'],
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'bbox_r_xmin': det['bbox_resized']['xmin'],
                    'bbox_r_ymin': det['bbox_resized']['ymin'],
                    'bbox_r_xmax': det['bbox_resized']['xmax'],
                    'bbox_r_ymax': det['bbox_resized']['ymax'],
                    'bbox_p_xmin': det['bbox_original_patch']['xmin'],
                    'bbox_p_ymin': det['bbox_original_patch']['ymin'],
                    'bbox_p_xmax': det['bbox_original_patch']['xmax'],
                    'bbox_p_ymax': det['bbox_original_patch']['ymax'],
                    'bbox_w_xmin': det['bbox_wsi']['xmin'],
                    'bbox_w_ymin': det['bbox_wsi']['ymin'],
                    'bbox_w_xmax': det['bbox_wsi']['xmax'],
                    'bbox_w_ymax': det['bbox_wsi']['ymax'],
                }
                detections_flat.append(det_flat)
            
            df_detections = pd.DataFrame(detections_flat)
            df_detections.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV: {csv_path}")
            
            annotated = overlay_detections_on_image(
                resized_img, boxes_resized, masks_resized, classes, confidences, config
            )
            
            annotated_path = config.VIS_DIR / f"{patch_file.stem}_annotated.png"
            cv2.imwrite(str(annotated_path), annotated)
            
            comparison_path = config.VIS_DIR / f"{patch_file.stem}_comparison.png"
            save_visualization_comparison(
                original_img, resized_img, annotated, patch_file.name, comparison_path
            )
            logger.info(f"Saved visualizations: {config.VIS_DIR}")
            
        else:
            logger.info("No detections found")
            patch_result = {
                'patch_name': patch_file.name,
                'patch_info': {
                    'original_width': original_width,
                    'original_height': original_height,
                    'wsi_xmin': patch_xmin,
                    'wsi_ymin': patch_ymin,
                    'wsi_xmax': int(patch_info['xmax']),
                    'wsi_ymax': int(patch_info['ymax']),
                },
                'num_detections': 0,
                'detections': []
            }
            all_results.append(patch_result)
    
    logger.info(f"\n{'='*80}")
    logger.info("SAVING CONSOLIDATED RESULTS")
    logger.info(f"{'='*80}")
    
    consolidated_json = config.OUTPUT_DIR / "all_patches_results.json"
    with open(consolidated_json, 'w') as f:
        json.dump({
            'wsi_info': {
                'name': config.WSI_NAME,
                'width': config.WSI_WIDTH,
                'height': config.WSI_HEIGHT,
            },
            'num_patches': len(all_results),
            'total_detections': sum(r['num_detections'] for r in all_results),
            'timestamp': datetime.now().isoformat(),
            'patches': all_results
        }, f, indent=2)
    logger.info(f"Consolidated JSON: {consolidated_json}")
    
    summary_stats = {
        'Total patches processed': len(all_results),
        'Patches with detections': sum(1 for r in all_results if r['num_detections'] > 0),
        'Total detections': sum(r['num_detections'] for r in all_results),
    }
    
    class_counts = {cls: 0 for cls in config.CLASS_NAMES}
    for result in all_results:
        for det in result['detections']:
            class_counts[det['class']] += 1
    
    summary_stats.update({f'{cls}_count': count for cls, count in class_counts.items()})
    
    summary_path = config.OUTPUT_DIR / "summary_statistics.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXTERNAL PATCH INFERENCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "="*80 + "\n")
    logger.info(f"Summary statistics: {summary_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("INFERENCE COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Visualizations: {config.VIS_DIR}")
    logger.info(f"Coordinates: {config.COORDS_DIR}")
    logger.info(f"{'='*80}\n")
    
    return all_results



def main():
    config = ExternalInferenceConfig()
    config.__post_init__()
    
    try:
        results = run_inference_on_external_patches(config)
        logger.info("Success!")
        return 0
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
