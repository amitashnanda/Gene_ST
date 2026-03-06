"""
DINOv3 Pretraining for Histopathology Images
============================================
This script pretrains a YOLOv11 backbone using DINOv3 distillation on unlabeled histopathological images.

Usage:
    python 1_dinov3_pretrain.py

Requirements:
    - lightly_train
    - ultralytics
    - torch with CUDA support
"""

import os
import sys
from pathlib import Path
import torch
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class Config:
    """Configuration for DINOv3 pretraining"""
    
  
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    DATA_DIR = BASE_DIR / "dataset_processed2/unlabeled/images"  
    OUTPUT_DIR = BASE_DIR / "models/dinov3_pretrained_3"
    CACHE_DIR = Path("/pscratch/sd/a/ananda/lightly-cache")  
    MODEL = "ultralytics/yolo11m-seg.yaml" 
    TEACHER = "dinov3/vitb16"
    TEACHER_URL = "https://dinov3.llamameta.net/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibnU1N3p4OTllbGR3aXowOTF5OGxjMnJpIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjQ5MDIwMjJ9fX1dfQ__&Signature=lFQHYTbqwTxxFTf1gvOzojMx2WtVES9pjpgY5FBtQgurCfBmnUoHts8ABj9iesCN0z%7EGC6QRh0%7Ezic-ePaLCCUed4WWwx6Ejz5zs6RIG0kg9AMmeTBwmYfEgpGxzeuyqWl-yVkjyXKrRFS%7EtP9YAAaNXGV0FWjxytceWhwVdTbZk-FgMIhCPQC8oA7W6eNW8F-lb4vrmYIDqsYVfr4pWxa8bm3Oj0jjMdYgNgq4Fq1qRTe6ZY9nZZuQAJSzMnMoI0gozbXOudizb4Ij5H8s8%7EcS2mfeNj7rkHkdY68qMPkxAAKt28tWdHN%7EeGX5raXLWZOTNyfpYrboraRXOh%7E%7Eftw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=2607216772981388"


    EPOCHS = 200  
    BATCH_SIZE = 16 
    NUM_WORKERS = 8  
    ACCELERATOR = "gpu"
    DEVICES = 4  
    METHOD = "distillation"
    

    SAVE_FREQ = 10  
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        if not self.DATA_DIR.exists():
            raise ValueError(f"Data directory not found: {self.DATA_DIR}")
        
       
        image_count = len(list(self.DATA_DIR.glob("*.png")))
        logger.info(f"Found {image_count} unlabeled images for pretraining")
        
        if image_count == 0:
            raise ValueError("No images found in data directory")
        
        
        total_batch_size = self.BATCH_SIZE * self.DEVICES
        steps_per_epoch = image_count // total_batch_size
        total_steps = steps_per_epoch * self.EPOCHS
        
        logger.info(f"Training statistics:")
        logger.info(f"  - Total batch size: {total_batch_size} (across {self.DEVICES} GPUs)")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total training steps: {total_steps}")
        logger.info(f"  - Estimated time: ~{self.EPOCHS * 2}-{self.EPOCHS * 3} minutes (2-3 min/epoch)")



def pretrain_dinov3():
    """
    Pretrain YOLOv11 backbone using DINOv3 distillation on unlabeled images.
    
    This function:
    1. Sets up the environment for multi-GPU training
    2. Configures DINOv3 teacher model
    3. Trains YOLOv11 student model via knowledge distillation
    4. Saves pretrained weights for downstream segmentation task
    """
    
  
    config = Config()
    config.__post_init__()
    os.environ["LIGHTLY_TRAIN_CACHE_DIR"] = str(config.CACHE_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
    

    logger.info("="*80)
    logger.info("DINOv3 PRETRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info(f"Model: {config.MODEL}")
    logger.info(f"Teacher: {config.TEACHER}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Batch size (per GPU): {config.BATCH_SIZE}")
    logger.info(f"Number of GPUs: {config.DEVICES}")
    logger.info(f"Number of workers (per GPU): {config.NUM_WORKERS}")
    logger.info("="*80)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU training required.")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs detected: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
       
        import lightly_train
        
    
        logger.info("Starting DINOv3 pretraining...")
        start_time = datetime.now()
        
        lightly_train.train(
            out=str(config.OUTPUT_DIR),
            data=str(config.DATA_DIR),
            model=config.MODEL,
            method=config.METHOD,
            method_args={
                "teacher": config.TEACHER,
                "teacher_url": config.TEACHER_URL,
            },
            accelerator=config.ACCELERATOR,
            devices=config.DEVICES,
            num_workers=config.NUM_WORKERS,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("PRETRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Pretrained weights saved to: {config.OUTPUT_DIR}/exported_models/")
        logger.info("="*80)
        exported_model = config.OUTPUT_DIR / "exported_models" / "exported_last.pt"
        if exported_model.exists():
            logger.info(f"✓ Exported model verified: {exported_model}")
            logger.info(f"  Model size: {exported_model.stat().st_size / (1024**2):.2f} MB")
        else:
            logger.warning(f"✗ Exported model not found at expected location: {exported_model}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import lightly_train: {e}")
        logger.error("Please install lightly_train: pip install lightly-train")
        return False
        
    except Exception as e:
        logger.error(f"Pretraining failed with error: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    logger.info("Starting DINOv3 pretraining script...")
    success = pretrain_dinov3()
    
    if success:
        logger.info("Script completed successfully")
        sys.exit(0)
    else:
        logger.error("Script failed")
        sys.exit(1)
