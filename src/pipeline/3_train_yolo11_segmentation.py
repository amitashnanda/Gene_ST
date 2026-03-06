"""
YOLOv11 Segmentation Training for Histopathology
=================================================
This script trains YOLOv11 segmentation model with DINOv3 pretrained weights.

Usage:
    python 3_train_yolo11_segmentation.py

Requirements:
    - ultralytics
    - torch with CUDA support
    - matplotlib
    - pandas
"""

import os
import sys
import yaml
import torch
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import json
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration for YOLOv11 segmentation training"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    PRETRAINED_WEIGHTS = BASE_DIR / "models/dinov3_pretrained_3/exported_models/exported_last.pt"
    DATA_YAML = BASE_DIR / "dataset_yolo2/data.yaml"
    OUTPUT_DIR = BASE_DIR / "runs"
    PLOTS_DIR = BASE_DIR / "plots/run_4"
    
    MODEL_SIZE = "yolo11m-seg"
    USE_PRETRAINED = True
    
    EPOCHS = 250
    BATCH_SIZE = 16
    IMAGE_SIZE = (768, 1024)
    
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 0.001
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    
    LR_SCHEDULER = "cosine"
    LRF = 0.01
    WARMUP_EPOCHS = 5
    WARMUP_MOMENTUM = 0.8
    WARMUP_BIAS_LR = 0.1
    
    BOX_LOSS_GAIN = 5.0
    CLS_LOSS_GAIN = 0.5
    DFL_LOSS_GAIN = 1.5
    
    SEG_LOSS_GAIN = 3.0
    
    USE_FOCAL_LOSS = False
    
    AUGMENTATION = True
    HSAUG_H = 0.0
    HSAUG_S = 0.0
    HSAUG_V = 0.0
    DEGREES = 90.0
    TRANSLATE = 0.2
    SCALE = 0.5
    SHEAR = 0.0
    PERSPECTIVE = 0.0
    FLIPUD = 0.5
    FLIPLR = 0.5
    MOSAIC = 1.0
    MIXUP = 0.1
    COPY_PASTE = 0.3
    
    DROPOUT = 0.1
    
    DEVICE = [0, 1, 2, 3]
    WORKERS = 8
    
    PATIENCE = 50
    SAVE_PERIOD = 10
    VERBOSE = True
    PLOTS = True
    
    VAL_INTERVAL = 1
    
    CLASS_WEIGHTS = None
    
    EXPERIMENT_NAME = f"yolo11m_seg_dinov3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.DATA_YAML.exists():
            raise ValueError(f"Data YAML not found: {self.DATA_YAML}")
        
        if self.USE_PRETRAINED and not self.PRETRAINED_WEIGHTS.exists():
            logger.warning(f"Pretrained weights not found: {self.PRETRAINED_WEIGHTS}")
            logger.warning("Will use COCO pretrained weights instead")
            self.USE_PRETRAINED = False
        
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. GPU training required.")
        
        available_gpus = torch.cuda.device_count()
        if len(self.DEVICE) > available_gpus:
            logger.warning(f"Requested {len(self.DEVICE)} GPUs but only {available_gpus} available")
            self.DEVICE = list(range(available_gpus))
        
        logger.info("="*80)
        logger.info("DATASET CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Labeled images for training: 242")
        logger.info(f"Test images: 35")
        logger.info(f"Classes: Crypt (617 instances), Gland (763 instances)")
        logger.info(f"Training statistics:")
        logger.info(f"  - Batch size: {self.BATCH_SIZE} (across {len(self.DEVICE)} GPUs)")
        logger.info(f"  - Steps per epoch: ~{242 // self.BATCH_SIZE}")
        logger.info(f"  - Total epochs: {self.EPOCHS}")
        logger.info(f"  - Total training steps: ~{(242 // self.BATCH_SIZE) * self.EPOCHS}")
        logger.info(f"  - Early stopping patience: {self.PATIENCE} epochs")
        logger.info("="*80)


def calculate_class_weights(data_yaml_path):
    """
    Calculate class weights based on instance counts in the dataset.
    
    Returns:
        list: Class weights [crypt_weight, gland_weight]
    """
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        info = data.get('dataset_info', {})
        crypt_count = info.get('total_crypt_instances', 1)
        gland_count = info.get('total_gland_instances', 1)
        
        total = crypt_count + gland_count
        
        crypt_weight = total / (2 * crypt_count)
        gland_weight = total / (2 * gland_count)
        
        logger.info(f"Calculated class weights: crypt={crypt_weight:.3f}, gland={gland_weight:.3f}")
        
        return [crypt_weight, gland_weight]
        
    except Exception as e:
        logger.warning(f"Could not calculate class weights: {e}")
        return [1.0, 1.0]

def plot_training_curves(results_csv, output_dir):
    """
    Plot training curves from results.csv
    
    Args:
        results_csv: Path to results.csv file
        output_dir: Directory to save plots
    """
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('YOLOv11 Segmentation Training Curves', fontsize=16, fontweight='bold')
        
        plots = [
            ('epoch', 'train/box_loss', 'Box Loss (Train)', axes[0, 0]),
            ('epoch', 'train/seg_loss', 'Segmentation Loss (Train)', axes[0, 1]),
            ('epoch', 'train/cls_loss', 'Classification Loss (Train)', axes[0, 2]),
            
            ('epoch', 'metrics/mAP50(B)', 'mAP@0.5 (Box)', axes[1, 0]),
            ('epoch', 'metrics/mAP50-95(B)', 'mAP@0.5:0.95 (Box)', axes[1, 1]),
            ('epoch', 'metrics/mAP50(M)', 'mAP@0.5 (Mask)', axes[1, 2]),
            
            ('epoch', 'metrics/precision(B)', 'Precision (Box)', axes[2, 0]),
            ('epoch', 'metrics/recall(B)', 'Recall (Box)', axes[2, 1]),
            ('epoch', 'lr/pg0', 'Learning Rate', axes[2, 2]),
        ]
        
        for x_col, y_col, title, ax in plots:
            if x_col in df.columns and y_col in df.columns:
                ax.plot(df[x_col], df[y_col], linewidth=2)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{title}\n(Data not available)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plot_path = output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to: {plot_path}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        loss_cols = ['train/box_loss', 'train/seg_loss', 'train/cls_loss', 'train/dfl_loss']
        available_losses = [col for col in loss_cols if col in df.columns]
        
        for col in available_losses:
            label = col.replace('train/', '').replace('_', ' ').title()
            ax.plot(df['epoch'], df[col], linewidth=2, label=label)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Losses Over Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_plot_path = output_dir / 'losses_comparison.png'
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Loss comparison plot saved to: {loss_plot_path}")
        
    except Exception as e:
        logger.error(f"Error plotting training curves: {e}")

def save_training_summary(results_csv, model_path, output_dir):
    """
    Save training summary with best metrics.
    """
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        
        if 'metrics/mAP50-95(M)' in df.columns:
            best_idx = df['metrics/mAP50-95(M)'].idxmax()
        elif 'metrics/mAP50-95(B)' in df.columns:
            best_idx = df['metrics/mAP50-95(B)'].idxmax()
        else:
            best_idx = df.index[-1]
        
        best_metrics = df.iloc[best_idx].to_dict()
        
        summary = {
            'training_completed': datetime.now().isoformat(),
            'model_path': str(model_path),
            'best_epoch': int(best_metrics.get('epoch', -1)),
            'best_metrics': {
                'mAP50_box': float(best_metrics.get('metrics/mAP50(B)', 0)),
                'mAP50-95_box': float(best_metrics.get('metrics/mAP50-95(B)', 0)),
                'mAP50_mask': float(best_metrics.get('metrics/mAP50(M)', 0)),
                'mAP50-95_mask': float(best_metrics.get('metrics/mAP50-95(M)', 0)),
                'precision_box': float(best_metrics.get('metrics/precision(B)', 0)),
                'recall_box': float(best_metrics.get('metrics/recall(B)', 0)),
            },
            'final_losses': {
                'box_loss': float(df.iloc[-1].get('train/box_loss', 0)),
                'seg_loss': float(df.iloc[-1].get('train/seg_loss', 0)),
                'cls_loss': float(df.iloc[-1].get('train/cls_loss', 0)),
                'dfl_loss': float(df.iloc[-1].get('train/dfl_loss', 0)),
            }
        }
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Best Epoch: {summary['best_epoch']}")
        logger.info(f"Best mAP@0.5:0.95 (Mask): {summary['best_metrics']['mAP50-95_mask']:.4f}")
        logger.info(f"Best mAP@0.5 (Mask): {summary['best_metrics']['mAP50_mask']:.4f}")
        logger.info(f"Best mAP@0.5:0.95 (Box): {summary['best_metrics']['mAP50-95_box']:.4f}")
        logger.info(f"Best Precision (Box): {summary['best_metrics']['precision_box']:.4f}")
        logger.info(f"Best Recall (Box): {summary['best_metrics']['recall_box']:.4f}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error saving training summary: {e}")


def train_yolo11():
    """
    Train YOLOv11 segmentation model with DINOv3 pretrained weights.
    """
    
    config = TrainingConfig()
    config.__post_init__()
    
    logger.info("="*80)
    logger.info("YOLOV11 SEGMENTATION TRAINING")
    logger.info("="*80)
    logger.info(f"Experiment: {config.EXPERIMENT_NAME}")
    logger.info(f"Model: {config.MODEL_SIZE}")
    logger.info(f"Use pretrained: {config.USE_PRETRAINED}")
    if config.USE_PRETRAINED:
        logger.info(f"Pretrained weights: {config.PRETRAINED_WEIGHTS}")
    logger.info(f"Data: {config.DATA_YAML}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Image size: {config.IMAGE_SIZE}")
    logger.info(f"Optimizer: {config.OPTIMIZER}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"GPUs: {config.DEVICE}")
    logger.info(f"Use focal loss: {config.USE_FOCAL_LOSS}")
    logger.info("="*80)
    
    if config.USE_FOCAL_LOSS:
        config.CLASS_WEIGHTS = calculate_class_weights(config.DATA_YAML)
    
    try:
        if config.USE_PRETRAINED:
            logger.info(f"Loading DINOv3 pretrained weights from: {config.PRETRAINED_WEIGHTS}")
            model = YOLO(str(config.PRETRAINED_WEIGHTS))
        else:
            logger.info(f"Loading {config.MODEL_SIZE} with COCO pretrained weights")
            model = YOLO(f"{config.MODEL_SIZE}.pt")
        
        logger.info("Starting training...")
        start_time = datetime.now()
        
        results = model.train(
            data=str(config.DATA_YAML),
            
            epochs=config.EPOCHS,
            batch=config.BATCH_SIZE,
            imgsz=config.IMAGE_SIZE,
            
            optimizer=config.OPTIMIZER,
            lr0=config.LEARNING_RATE,
            lrf=config.LRF,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY,
            warmup_epochs=config.WARMUP_EPOCHS,
            warmup_momentum=config.WARMUP_MOMENTUM,
            warmup_bias_lr=config.WARMUP_BIAS_LR,
            
            box=config.BOX_LOSS_GAIN,
            cls=config.CLS_LOSS_GAIN,
            dfl=config.DFL_LOSS_GAIN,
            
            
            hsv_h=config.HSAUG_H,
            hsv_s=config.HSAUG_S,
            hsv_v=config.HSAUG_V,
            degrees=config.DEGREES,
            translate=config.TRANSLATE,
            scale=config.SCALE,
            shear=config.SHEAR,
            perspective=config.PERSPECTIVE,
            flipud=config.FLIPUD,
            fliplr=config.FLIPLR,
            mosaic=config.MOSAIC,
            mixup=config.MIXUP,
            copy_paste=config.COPY_PASTE,
            
            dropout=config.DROPOUT,
            
            overlap_mask=True,
            mask_ratio=4,
            
            device=config.DEVICE,
            workers=config.WORKERS,
            
            project=str(config.OUTPUT_DIR),
            name=config.EXPERIMENT_NAME,
            exist_ok=True,
            
            patience=config.PATIENCE,
            save=True,
            save_period=config.SAVE_PERIOD,
            verbose=config.VERBOSE,
            plots=config.PLOTS,
            val=True,
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Results saved to: {config.OUTPUT_DIR / config.EXPERIMENT_NAME}")
        logger.info("="*80)
        
        results_csv = config.OUTPUT_DIR / config.EXPERIMENT_NAME / "results.csv"
        if results_csv.exists():
            plot_training_curves(results_csv, config.PLOTS_DIR)
            
            best_model = config.OUTPUT_DIR / config.EXPERIMENT_NAME / "weights" / "best.pt"
            if best_model.exists():
                final_model_dir = config.BASE_DIR / "models" / "yolo11_final"
                final_model_dir.mkdir(parents=True, exist_ok=True)
                final_model_path = final_model_dir / "best.pt"
                shutil.copy2(best_model, final_model_path)
                logger.info(f"Best model copied to: {final_model_path}")
                
                save_training_summary(results_csv, final_model_path, config.PLOTS_DIR)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    logger.info("Starting YOLOv11 segmentation training script...")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    success = train_yolo11()
    
    if success:
        logger.info("Training completed successfully")
        sys.exit(0)
    else:
        logger.error("Training failed")
        sys.exit(1)
