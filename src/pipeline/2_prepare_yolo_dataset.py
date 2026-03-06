"""
YOLO Dataset Preparation for Train_Roboflow Dataset
===================================================
This script splits Train_Roboflow labeled data (with .txt annotations) into train/val/test sets
and creates YOLO dataset structure in dataset_yolo2 folder.

Input:
    dataset_processed2/labeled/
        images/     (1024x768 images)
        labels/     (.txt YOLO format annotations)

Output:
    dataset_yolo2/
        train/
            images/
            labels/
        val/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml

Usage:
    python prepare_yolo_dataset2.py

Requirements:
    - PyYAML
    - scikit-learn
"""

import os
import sys
import yaml
import shutil
import logging
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for YOLO dataset preparation"""
    
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    INPUT_DIR = BASE_DIR / "dataset_processed2/labeled"
    OUTPUT_DIR = BASE_DIR / "dataset_yolo2"
    
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.25
    TEST_RATIO = 0.05
    
    CLASS_NAMES = ["crypt", "gland"]
    NUM_CLASSES = 2
    
    IMAGE_SIZE = (1024, 768)
    
    RANDOM_SEED = 42
    
    def __post_init__(self):
        """Validate configuration"""
        assert abs(self.TRAIN_RATIO + self.VAL_RATIO + self.TEST_RATIO - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        if not self.INPUT_DIR.exists():
            raise ValueError(f"Input directory not found: {self.INPUT_DIR}")
        
        images_dir = self.INPUT_DIR / "images"
        labels_dir = self.INPUT_DIR / "labels"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        if not labels_dir.exists():
            raise ValueError(f"Labels directory not found: {labels_dir}")


def collect_image_label_pairs(input_dir):
    """
    Collect all image and corresponding label file pairs.
    
    Returns:
        list: [(image_path, label_path), ...]
    """
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    
    image_files = []
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
        image_files.extend(images_dir.glob(ext))
    
    logger.info(f"Found {len(image_files)} images")
    
    pairs = []
    missing_labels = []
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)
    
    logger.info(f"Found {len(pairs)} image-label pairs")
    
    if missing_labels:
        logger.warning(f"⚠ {len(missing_labels)} images without labels:")
        for name in missing_labels[:10]:
            logger.warning(f"  - {name}")
        if len(missing_labels) > 10:
            logger.warning(f"  ... and {len(missing_labels) - 10} more")
    
    return pairs


def analyze_annotations(pairs):
    """
    Analyze annotation statistics from label files.
    
    Returns:
        dict: Statistics about class distribution
    """
    stats = {
        'total_instances': 0,
        'crypt_instances': 0,
        'gland_instances': 0,
        'images_with_crypts': 0,
        'images_with_glands': 0,
        'images_with_both': 0,
    }
    
    for img_path, label_path in pairs:
        has_crypt = False
        has_gland = False
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 1:
                    continue
                
                class_id = int(parts[0])
                stats['total_instances'] += 1
                
                if class_id == 0:
                    stats['crypt_instances'] += 1
                    has_crypt = True
                elif class_id == 1:
                    stats['gland_instances'] += 1
                    has_gland = True
        
        if has_crypt:
            stats['images_with_crypts'] += 1
        if has_gland:
            stats['images_with_glands'] += 1
        if has_crypt and has_gland:
            stats['images_with_both'] += 1
    
    return stats


def create_yolo_dataset(config):
    """
    Create YOLO format dataset by splitting Train_Roboflow data.
    """
    logger.info("="*80)
    logger.info("YOLO DATASET PREPARATION - TRAIN_ROBOFLOW")
    logger.info("="*80)
    
    pairs = collect_image_label_pairs(config.INPUT_DIR)
    
    if len(pairs) == 0:
        raise ValueError("No image-label pairs found!")
    
    logger.info("\nAnalyzing annotations...")
    annotation_stats = analyze_annotations(pairs)
    
    logger.info(f"Total instances: {annotation_stats['total_instances']}")
    logger.info(f"  Crypt instances: {annotation_stats['crypt_instances']}")
    logger.info(f"  Gland instances: {annotation_stats['gland_instances']}")
    logger.info(f"Images with crypts: {annotation_stats['images_with_crypts']}")
    logger.info(f"Images with glands: {annotation_stats['images_with_glands']}")
    logger.info(f"Images with both: {annotation_stats['images_with_both']}")
    
    logger.info("\nSplitting dataset...")
    
    train_val_pairs, test_pairs = train_test_split(
        pairs,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_SEED
    )
    
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=val_ratio_adjusted,
        random_state=config.RANDOM_SEED
    )
    
    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_pairs)} images ({len(train_pairs)/len(pairs)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/len(pairs)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/len(pairs)*100:.1f}%)")
    
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    for split in ['train', 'val', 'test']:
        (config.OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    split_stats = defaultdict(lambda: {'images': 0, 'crypt_instances': 0, 'gland_instances': 0})
    
    for split_name, split_pairs in splits.items():
        logger.info(f"\nProcessing {split_name} set...")
        
        for img_path, label_path in split_pairs:
            img_dst = config.OUTPUT_DIR / split_name / 'images' / img_path.name
            shutil.copy2(img_path, img_dst)
            
            label_dst = config.OUTPUT_DIR / split_name / 'labels' / label_path.name
            shutil.copy2(label_path, label_dst)
            
            split_stats[split_name]['images'] += 1
            
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 1:
                        continue
                    
                    class_id = int(parts[0])
                    if class_id == 0:
                        split_stats[split_name]['crypt_instances'] += 1
                    elif class_id == 1:
                        split_stats[split_name]['gland_instances'] += 1
        
        logger.info(f"  {split_name}: {split_stats[split_name]['images']} images, "
                   f"{split_stats[split_name]['crypt_instances']} crypts, "
                   f"{split_stats[split_name]['gland_instances']} glands")
    
    create_data_yaml(config, split_stats, annotation_stats)
    
    logger.info("="*80)
    logger.info("DATASET PREPARATION COMPLETED")
    logger.info("="*80)
    
    return split_stats, annotation_stats


def create_data_yaml(config, split_stats, annotation_stats):
    """Create YOLO data configuration file."""
    
    data_yaml = {
        'path': str(config.OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': config.NUM_CLASSES,
        'names': config.CLASS_NAMES,
        
        'dataset_info': {
            'description': 'Train_Roboflow Histopathology Segmentation (1024x768)',
            'source': 'Train_Roboflow dataset',
            'created': datetime.now().isoformat(),
            'image_size': f"{config.IMAGE_SIZE[0]}x{config.IMAGE_SIZE[1]}",
            'normalization': 'Reinhard',
            'train_images': split_stats['train']['images'],
            'val_images': split_stats['val']['images'],
            'test_images': split_stats['test']['images'],
            'total_crypt_instances': annotation_stats['crypt_instances'],
            'total_gland_instances': annotation_stats['gland_instances'],
            'train_crypts': split_stats['train']['crypt_instances'],
            'train_glands': split_stats['train']['gland_instances'],
            'val_crypts': split_stats['val']['crypt_instances'],
            'val_glands': split_stats['val']['gland_instances'],
            'test_crypts': split_stats['test']['crypt_instances'],
            'test_glands': split_stats['test']['gland_instances'],
        }
    }
    
    yaml_path = config.OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\nCreated data.yaml at: {yaml_path}")
    
    logger.info("\nDataset configuration:")
    logger.info(yaml.dump(data_yaml, default_flow_style=False, sort_keys=False))



def main():
    """Main execution function."""
    
    try:
        config = DatasetConfig()
        config.__post_init__()
        
        split_stats, annotation_stats = create_yolo_dataset(config)
        
        logger.info("\n" + "="*80)
        logger.info("FINAL STATISTICS")
        logger.info("="*80)
        
        total_images = sum(s['images'] for s in split_stats.values())
        total_crypts = annotation_stats['crypt_instances']
        total_glands = annotation_stats['gland_instances']
        
        logger.info(f"Total images: {total_images}")
        logger.info(f"Total crypt instances: {total_crypts}")
        logger.info(f"Total gland instances: {total_glands}")
        logger.info(f"Class ratio (crypt:gland): {total_crypts}:{total_glands} "
                   f"({total_crypts/(total_crypts+total_glands)*100:.1f}%:"
                   f"{total_glands/(total_crypts+total_glands)*100:.1f}%)")
        
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATIONS")
        logger.info("="*80)
        
        if total_crypts < total_glands * 0.5 or total_glands < total_crypts * 0.5:
            logger.warning("⚠ Significant class imbalance detected!")
            logger.info("  → Recommend using focal loss or weighted loss")
            logger.info("  → Consider class-balanced sampling during training")
        else:
            logger.info("✓ Class distribution is relatively balanced")
        
        if total_images < 500:
            logger.warning("⚠ Moderate dataset size detected!")
            logger.info("  → Recommend using data augmentation")
            logger.info("  → Consider using pretrained weights (DINOv3)")
        else:
            logger.info("✓ Dataset size is adequate")
        
        logger.info("\n✓ Dataset ready for YOLO training!")
        logger.info(f"  Dataset location: {config.OUTPUT_DIR}")
        logger.info(f"  Data config: {config.OUTPUT_DIR / 'data.yaml'}")
        
        logger.info("\nNext steps:")
        logger.info("  1. Update training script to use dataset_yolo2/data.yaml")
        logger.info("  2. Adjust batch size and epochs based on dataset size")
        logger.info("  3. Run DINOv3 pretraining on unlabeled data if needed")
        
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    logger.info("Starting YOLO dataset preparation for Train_Roboflow...")
    success = main()
    
    if success:
        logger.info("Dataset preparation completed successfully")
        sys.exit(0)
    else:
        logger.error("Dataset preparation failed")
        sys.exit(1)
