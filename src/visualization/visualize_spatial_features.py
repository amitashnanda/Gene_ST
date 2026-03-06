"""
Spatial Feature Visualization for DINOv3-YOLO11 Pretrained Model
================================================================

This script extracts SPATIAL feature maps from intermediate YOLO11 backbone layers
to visualize what features the model learned for u-shape and o-shape structures
WITHIN each image (not just global features).

Key capabilities:
1. Load pretrained YOLO11 checkpoint
2. Extract spatial feature maps from multiple backbone layers (e.g., 20x20, 40x40 grids)
3. Apply PCA to spatial features to create attention heatmaps
4. Visualize which regions of the image activate different features
5. Compare feature activations between different spatial regions

Usage:
    # Visualize random samples
    python visualize_spatial_features.py --n_samples 5
    
    # Visualize specific image
    python visualize_spatial_features.py --image_path /path/to/image.png
    
    # Extract features from specific layers
    python visualize_spatial_features.py --layers 6 9 12 --n_samples 3

Author: Ananda
Date: December 2024
"""

import os
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as T

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



class Config:
    BASE_DIR = Path("/pscratch/sd/a/ananda/Spatial/Final_Code")
    DATA_DIR = BASE_DIR / "dataset_processed/unlabeled/images"
    EXPERIMENT_DIR = BASE_DIR / "models/dinov3_pretrained_3"
    
    CHECKPOINT = EXPERIMENT_DIR / "checkpoints" / "last.ckpt"
    VIS_DIR = EXPERIMENT_DIR / "spatial_feature_visualizations"
    
    TARGET_LAYERS = [6, 9, 12, 15]
    
    INPUT_SIZE = (640, 640)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        self.VIS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualization output directory: {self.VIS_DIR}")



class FeatureExtractor(nn.Module):
    """
    Extract intermediate feature maps from YOLO11 backbone.
    
    This class hooks into specified layers of the YOLO backbone to extract
    spatial feature maps during forward pass.
    """
    
    def __init__(self, model, target_layers: List[int]):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self.features: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on target layers"""
        
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.features[layer_idx] = output.detach()
            return hook
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            backbone = self.model.model.model
        elif hasattr(self.model, 'model'):
            backbone = self.model.model
        else:
            backbone = self.model
        
        for layer_idx in self.target_layers:
            if layer_idx < len(backbone):
                hook = backbone[layer_idx].register_forward_hook(get_hook(layer_idx))
                self.hooks.append(hook)
                logger.info(f"Registered hook on layer {layer_idx}: {type(backbone[layer_idx]).__name__}")
            else:
                logger.warning(f"Layer {layer_idx} out of range (model has {len(backbone)} layers)")
    
    def forward(self, x):
        """Forward pass and collect features"""
        self.features = {}
        _ = self.model(x)
        return self.features
    
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []



def load_pretrained_model(checkpoint_path: Path, device: str = "cuda") -> nn.Module:
    """
    Load the pretrained YOLO11 model from lightly checkpoint.
    
    The lightly checkpoint contains the model state dict with specific structure.
    We need to load it properly to access the backbone.
    """
    logger.info(f"Loading pretrained model from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("Found 'state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            logger.info("Using checkpoint directly as state_dict")
        
        from ultralytics import YOLO
        
        model = YOLO("yolo11m-seg.pt")
        
        model_state = model.model.state_dict()
        
        pretrained_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.model.', '').replace('model.', '')
            
            if new_key in model_state:
                pretrained_dict[new_key] = v
        
        logger.info(f"Loading {len(pretrained_dict)}/{len(model_state)} parameters from checkpoint")
        
        model.model.load_state_dict(pretrained_dict, strict=False)
        model.model.eval()
        model.model.to(device)
        
        logger.info("✓ Model loaded successfully")
        return model.model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.exception("Full traceback:")
        raise



def load_and_preprocess_image(image_path: Path, target_size: Tuple[int, int] = (640, 640)) -> Tuple[Image.Image, torch.Tensor]:
    """Load image and convert to tensor"""
    img = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    return img, img_tensor



def apply_pca_to_spatial_features(features: torch.Tensor, n_components: int = 3) -> np.ndarray:
    """
    Apply PCA to spatial features to reduce channel dimension.
    
    Args:
        features: Tensor of shape (B, C, H, W) where C is number of channels
        n_components: Number of PCA components (typically 3 for RGB visualization)
    
    Returns:
        pca_features: Array of shape (H, W, n_components)
    """
    B, C, H, W = features.shape
    
    features_flat = features[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_flat)
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_features_flat = pca.fit_transform(features_scaled)
    
    pca_features = pca_features_flat.reshape(H, W, n_components)
    
    for i in range(n_components):
        pca_features[:, :, i] = (pca_features[:, :, i] - pca_features[:, :, i].min()) / \
                                 (pca_features[:, :, i].max() - pca_features[:, :, i].min() + 1e-8)
    
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%} "
                f"(components: {pca.explained_variance_ratio_})")
    
    return pca_features, pca


def create_attention_heatmap(features: torch.Tensor) -> np.ndarray:
    """
    Create attention heatmap by computing spatial importance.
    
    Uses L2 norm across channels for each spatial location.
    """
    attention = torch.norm(features[0], dim=0).cpu().numpy()
    
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    return attention


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, 
                             alpha: float = 0.5, colormap: str = 'jet') -> np.ndarray:
    """Overlay heatmap on original image"""
    img_array = np.array(image)
    H, W = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    img_array = img_array.astype(np.float32)
    heatmap_colored = heatmap_colored.astype(np.float32)
    blended = (1 - alpha) * img_array + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended



def visualize_single_image_features(
    image_path: Path,
    model: nn.Module,
    target_layers: List[int],
    output_dir: Path,
    device: str = "cuda"
):
    """
    Create comprehensive visualization of spatial features for one image.
    
    This shows:
    1. Original image
    2. PCA feature maps from multiple layers (as RGB images)
    3. Attention heatmaps overlaid on image
    4. Individual PCA components as heatmaps
    """
    logger.info(f"Processing image: {image_path.name}")
    
    img, img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    feature_extractor = FeatureExtractor(model, target_layers)
    with torch.no_grad():
        features_dict = feature_extractor(img_tensor)
    
    n_layers = len(target_layers)
    fig = plt.figure(figsize=(24, 4 * n_layers))
    gs = fig.add_gridspec(n_layers, 6, hspace=0.3, wspace=0.3)
    
    for idx, layer_idx in enumerate(target_layers):
        if layer_idx not in features_dict:
            logger.warning(f"Layer {layer_idx} not found in extracted features")
            continue
        
        features = features_dict[layer_idx]
        B, C, H, W = features.shape
        logger.info(f"Layer {layer_idx}: shape {features.shape} ({C} channels, {H}x{W} spatial)")
        
        row = idx
        
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(img)
        ax1.set_title(f"Layer {layer_idx}\nOriginal Image", fontsize=10, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[row, 1])
        pca_features, pca = apply_pca_to_spatial_features(features, n_components=3)
        ax2.imshow(pca_features)
        ax2.set_title(f"PCA Features (RGB)\n{H}x{W} spatial grid\n{C} channels", 
                     fontsize=10, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[row, 2])
        attention = create_attention_heatmap(features)
        blended = overlay_heatmap_on_image(img, attention, alpha=0.5, colormap='jet')
        ax3.imshow(blended)
        ax3.set_title(f"Attention Heatmap\n(L2 norm overlay)", fontsize=10, fontweight='bold')
        ax3.axis('off')
        
        for comp_idx in range(3):
            ax = fig.add_subplot(gs[row, 3 + comp_idx])
            pca_comp = pca_features[:, :, comp_idx]
            
            pca_comp_resized = cv2.resize(pca_comp, (img.size[0], img.size[1]), 
                                         interpolation=cv2.INTER_LINEAR)
            
            im = ax.imshow(pca_comp_resized, cmap='viridis')
            var = pca.explained_variance_ratio_[comp_idx]
            ax.set_title(f"PC{comp_idx+1}\n({var:.1%} var)", fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f"Spatial Feature Analysis: {image_path.name}\n"
                 f"Shows what features the DINOv3-YOLO11 model learned at different layers",
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / f"spatial_features_{image_path.stem}.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    logger.info(f"✓ Saved visualization: {output_path}")
    plt.close()
    
    feature_extractor.remove_hooks()


def visualize_pca_comparison(
    image_path: Path,
    model: nn.Module,
    target_layers: List[int],
    output_dir: Path,
    device: str = "cuda"
):
    """
    Create a comparison of PCA visualizations across all layers in one figure.
    This helps see how features evolve from early to late layers.
    """
    logger.info(f"Creating layer comparison for: {image_path.name}")
    
    img, img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    feature_extractor = FeatureExtractor(model, target_layers)
    with torch.no_grad():
        features_dict = feature_extractor(img_tensor)
    
    n_layers = len(target_layers)
    fig, axes = plt.subplots(2, n_layers + 1, figsize=(4 * (n_layers + 1), 8))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for idx, layer_idx in enumerate(target_layers):
        if layer_idx not in features_dict:
            continue
        
        features = features_dict[layer_idx]
        B, C, H, W = features.shape
        
        col = idx + 1
        
        pca_features, pca = apply_pca_to_spatial_features(features, n_components=3)
        axes[0, col].imshow(pca_features)
        axes[0, col].set_title(f"Layer {layer_idx}\n{H}x{W}, {C}ch", 
                              fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
        
        attention = create_attention_heatmap(features)
        blended = overlay_heatmap_on_image(img, attention, alpha=0.5)
        axes[1, col].imshow(blended)
        axes[1, col].set_title(f"Attention Map", fontsize=10)
        axes[1, col].axis('off')
    
    fig.suptitle(f"Layer-by-Layer Feature Evolution: {image_path.name}\n"
                 f"Top: PCA features | Bottom: Attention maps",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f"layer_comparison_{image_path.stem}.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    logger.info(f"✓ Saved layer comparison: {output_path}")
    plt.close()
    
    feature_extractor.remove_hooks()



def main():
    parser = argparse.ArgumentParser(
        description="Visualize spatial features learned by DINOv3-YOLO11 pretrained model"
    )
    parser.add_argument("--image_path", type=str, default=None,
                       help="Path to specific image to visualize")
    parser.add_argument("--n_samples", type=int, default=5,
                       help="Number of random samples to visualize (if no image_path)")
    parser.add_argument("--layers", nargs='+', type=int, default=[6, 9, 12, 15],
                       help="Which backbone layers to extract features from")
    
    args = parser.parse_args()
    
    cfg = Config()
    cfg.__post_init__()
    cfg.TARGET_LAYERS = args.layers
    
    logger.info("="*80)
    logger.info("SPATIAL FEATURE VISUALIZATION - DINOv3-YOLO11")
    logger.info("="*80)
    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Target layers: {cfg.TARGET_LAYERS}")
    logger.info(f"Checkpoint: {cfg.CHECKPOINT}")
    logger.info("="*80)
    
    model = load_pretrained_model(cfg.CHECKPOINT, cfg.DEVICE)
    
    if args.image_path:
        image_paths = [Path(args.image_path)]
        if not image_paths[0].exists():
            logger.error(f"Image not found: {args.image_path}")
            return
    else:
        all_images = list(cfg.DATA_DIR.glob("*.png"))
        if len(all_images) == 0:
            logger.error(f"No images found in {cfg.DATA_DIR}")
            return
        
        n_samples = min(args.n_samples, len(all_images))
        image_paths = np.random.choice(all_images, size=n_samples, replace=False)
        logger.info(f"Selected {n_samples} random images for visualization")
    
    for image_path in image_paths:
        try:
            visualize_single_image_features(
                image_path, model, cfg.TARGET_LAYERS, cfg.VIS_DIR, cfg.DEVICE
            )
            
            visualize_pca_comparison(
                image_path, model, cfg.TARGET_LAYERS, cfg.VIS_DIR, cfg.DEVICE
            )
            
        except Exception as e:
            logger.error(f"Failed to visualize {image_path.name}: {e}")
            logger.exception("Full traceback:")
            continue
    
    logger.info("="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info(f"Output directory: {cfg.VIS_DIR}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("📊 WHAT TO LOOK FOR IN THE VISUALIZATIONS:")
    print("="*80)
    print("\n1. PCA Features (RGB images):")
    print("   - Different colors = different features activated")
    print("   - If u-shapes show one color pattern, o-shapes show another = GOOD")
    print("   - Similar colors everywhere = model didn't learn spatial differences")
    
    print("\n2. Attention Heatmaps:")
    print("   - Hot colors (red/yellow) = regions model focuses on")
    print("   - Should highlight u-shape and o-shape boundaries if learned well")
    
    print("\n3. Individual PCA Components (PC1, PC2, PC3):")
    print("   - PC1 might activate on u-shapes, PC2 on o-shapes")
    print("   - Or different PCs capture different structural patterns")
    
    print("\n4. Layer Evolution:")
    print("   - Early layers (6): edges, textures")
    print("   - Middle layers (9, 12): shape patterns (u-shape vs o-shape)")
    print("   - Late layers (15): context and object-level features")
    
    print("\n" + "="*80)
    print(f"💡 All visualizations saved to: {cfg.VIS_DIR}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
