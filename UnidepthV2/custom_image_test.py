#!/usr/bin/env python3
"""
Test UniDepthV2 on custom images
Usage: python custom_image_test.py <image_path>
Example: python custom_image_test.py my_image.jpg
"""

import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from unidepth.models import UniDepthV2
from unidepth.utils import colorize


def test_custom_image(image_path):
    """Test UniDepthV2 on a custom image"""
    
    print(f"Testing UniDepthV2 on: {image_path}")
    
    # Load UniDepthV2 model
    print("Loading UniDepthV2 model...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device).eval()
    
    # Load custom image
    print("Loading custom image...")
    try:
        rgb = np.array(Image.open(image_path))
        print(f"Image loaded: {rgb.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Convert to tensor format
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model.infer(rgb_torch)
    
    # Extract results
    depth = predictions["depth"].squeeze().cpu().numpy()
    confidence = predictions.get("confidence", None)
    if confidence is not None:
        confidence = confidence.squeeze().cpu().numpy()
    
    print(f"Available predictions: {list(predictions.keys())}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Predicted depth
    depth_col = colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
    axes[1].imshow(depth_col)
    axes[1].set_title("Predicted Depth")
    axes[1].axis('off')
    
    # Confidence map (if available)
    if confidence is not None:
        axes[2].imshow(confidence, cmap="viridis")
        axes[2].set_title("Confidence Map")
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, "No confidence\navailable", 
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("Confidence Map")
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_name = f"custom_test_{image_path.split('/')[-1].split('.')[0]}.png"
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"Results saved to: {output_name}")
    
    # Save individual outputs
    Image.fromarray(depth_col).save(f"custom_depth_{image_path.split('/')[-1].split('.')[0]}.png")
    if confidence is not None:
        Image.fromarray((confidence * 255).astype(np.uint8)).save(f"custom_confidence_{image_path.split('/')[-1].split('.')[0]}.png")
    
    print("Individual outputs saved:")
    print(f"- custom_depth_{image_path.split('/')[-1].split('.')[0]}.png")
    if confidence is not None:
        print(f"- custom_confidence_{image_path.split('/')[-1].split('.')[0]}.png")
    
    return predictions


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python custom_image_test.py <image_path>")
        print("Example: python custom_image_test.py my_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    predictions = test_custom_image(image_path)
