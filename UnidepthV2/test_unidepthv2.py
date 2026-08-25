#!/usr/bin/env python3
"""
Simple test script for UniDepthV2
Usage: python test_unidepthv2.py
"""

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from unidepth.models import UniDepthV2
from unidepth.utils import colorize


def test_unidepthv2():
    """Test UniDepthV2 on a sample image"""
    
    print("Loading UniDepthV2 model...")
    # Load UniDepthV2 with ViT-L backbone (you can also use 'vits14' or 'vitb14')
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device).eval()
    
    # Load the demo image
    print("Loading demo image...")
    rgb_path = "assets/demo/rgb.png"
    rgb = np.array(Image.open(rgb_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    # Load intrinsics if available
    intrinsics_path = "assets/demo/intrinsics.npy"
    try:
        intrinsics = torch.from_numpy(np.load(intrinsics_path))
        print("Using provided intrinsics")
    except:
        print("No intrinsics provided, model will predict them")
        intrinsics = None
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        if intrinsics is not None:
            predictions = model.infer(rgb_torch.squeeze(0), intrinsics)
        else:
            predictions = model.infer(rgb_torch.squeeze(0))
    
    # Extract results
    depth = predictions["depth"].squeeze().cpu().numpy()
    confidence = predictions.get("confidence", None)
    if confidence is not None:
        confidence = confidence.squeeze().cpu().numpy()
    
    # Load ground truth for comparison
    depth_gt = np.array(Image.open("assets/demo/depth.png")).astype(float) / 1000.0
    
    # Calculate metrics
    depth_arel = np.abs(depth_gt - depth) / depth_gt
    depth_arel[depth_gt == 0.0] = 0.0
    arel_mean = depth_arel[depth_gt > 0].mean() * 100
    
    print(f"ARel: {arel_mean:.2f}%")
    print(f"Available predictions: {list(predictions.keys())}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Original RGB")
    axes[0, 0].axis('off')
    
    # Ground truth depth
    depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
    axes[0, 1].imshow(depth_gt_col)
    axes[0, 1].set_title("Ground Truth Depth")
    axes[0, 1].axis('off')
    
    # Predicted depth
    depth_pred_col = colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
    axes[0, 2].imshow(depth_pred_col)
    axes[0, 2].set_title("Predicted Depth")
    axes[0, 2].axis('off')
    
    # Error map
    depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")
    axes[1, 0].imshow(depth_error_col)
    axes[1, 0].set_title("Depth Error (ARel)")
    axes[1, 0].axis('off')
    
    # Confidence map (if available)
    if confidence is not None:
        axes[1, 1].imshow(confidence, cmap="viridis")
        axes[1, 1].set_title("Confidence Map")
        axes[1, 1].axis('off')
    else:
        axes[1, 1].text(0.5, 0.5, "No confidence\navailable", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Confidence Map")
        axes[1, 1].axis('off')
    
    # Depth histogram
    valid_depths = depth[depth > 0]
    axes[1, 2].hist(valid_depths.flatten(), bins=50, alpha=0.7, color='blue', label='Predicted')
    axes[1, 2].hist(depth_gt[depth_gt > 0].flatten(), bins=50, alpha=0.7, color='red', label='Ground Truth')
    axes[1, 2].set_xlabel("Depth (m)")
    axes[1, 2].set_ylabel("Frequency")
    axes[1, 2].set_title("Depth Distribution")
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig("unidepthv2_test_results.png", dpi=150, bbox_inches='tight')
    print("Results saved to: unidepthv2_test_results.png")
    
    # Save individual outputs
    Image.fromarray(depth_pred_col).save("predicted_depth.png")
    Image.fromarray(depth_error_col).save("depth_error.png")
    if confidence is not None:
        Image.fromarray((confidence * 255).astype(np.uint8)).save("confidence.png")
    
    print("Individual outputs saved:")
    print("- predicted_depth.png")
    print("- depth_error.png")
    if confidence is not None:
        print("- confidence.png")
    
    return predictions


if __name__ == "__main__":
    predictions = test_unidepthv2()
