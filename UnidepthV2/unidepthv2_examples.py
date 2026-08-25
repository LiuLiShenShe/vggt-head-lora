#!/usr/bin/env python3
"""
Comprehensive examples for UniDepthV2
Shows different models, configurations, and usage patterns
"""

import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2
from unidepth.utils import colorize
from unidepth.utils.camera import Pinhole, Fisheye624


def example_1_basic_usage():
    """Basic usage with default settings"""
    print("=== Example 1: Basic Usage ===")
    
    # Load model
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Load image
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Inference
    predictions = model.infer(rgb_torch)
    
    print(f"Available outputs: {list(predictions.keys())}")
    print(f"Depth shape: {predictions['depth'].shape}")
    print(f"Confidence shape: {predictions['confidence'].shape}")
    
    return predictions


def example_2_different_models():
    """Show different UniDepthV2 model variants"""
    print("\n=== Example 2: Different Models ===")
    
    models = {
        "ViT-S": "lpiccinelli/unidepth-v2-vits14",
        "ViT-B": "lpiccinelli/unidepth-v2-vitb14", 
        "ViT-L": "lpiccinelli/unidepth-v2-vitl14"
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, model_id in models.items():
        print(f"\nLoading {name} model...")
        model = UniDepthV2.from_pretrained(model_id)
        model = model.to(device).eval()
        
        # Load image
        rgb = np.array(Image.open("assets/demo/rgb.png"))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
        
        # Inference
        with torch.no_grad():
            predictions = model.infer(rgb_torch)
        
        depth = predictions["depth"].squeeze().cpu().numpy()
        print(f"{name} - Depth range: {depth.min():.3f}m to {depth.max():.3f}m")


def example_3_with_intrinsics():
    """Using UniDepthV2 with known camera intrinsics"""
    print("\n=== Example 3: With Camera Intrinsics ===")
    
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Load image and intrinsics
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    intrinsics = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    
    # Create camera object
    camera = Pinhole(K=intrinsics)
    
    # Inference with camera
    predictions = model.infer(rgb_torch, camera)
    
    print(f"Using Pinhole camera with intrinsics")
    print(f"Intrinsics matrix:\n{intrinsics}")
    print(f"Depth shape: {predictions['depth'].shape}")


def example_4_resolution_levels():
    """Demonstrate different resolution levels"""
    print("\n=== Example 4: Resolution Levels ===")
    
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Load image
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Test different resolution levels
    resolution_levels = [1, 5, 9]
    
    for level in resolution_levels:
        print(f"\nTesting resolution level {level}")
        model.resolution_level = level
        
        with torch.no_grad():
            predictions = model.infer(rgb_torch)
        
        depth = predictions["depth"].squeeze().cpu().numpy()
        print(f"Resolution level {level} - Depth shape: {depth.shape}")


def example_5_interpolation_modes():
    """Show different interpolation modes"""
    print("\n=== Example 5: Interpolation Modes ===")
    
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Load image
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    interpolation_modes = ["nearest", "bilinear", "bicubic"]
    
    for mode in interpolation_modes:
        print(f"\nTesting interpolation mode: {mode}")
        model.interpolation_mode = mode
        
        with torch.no_grad():
            predictions = model.infer(rgb_torch)
        
        depth = predictions["depth"].squeeze().cpu().numpy()
        print(f"{mode} - Depth range: {depth.min():.3f}m to {depth.max():.3f}m")


def example_6_camera_types():
    """Demonstrate different camera types"""
    print("\n=== Example 6: Camera Types ===")
    
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Load image
    rgb = np.array(Image.open("assets/demo/rgb.png"))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
    
    # Pinhole camera (default)
    print("\nPinhole camera:")
    intrinsics = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
    camera_pinhole = Pinhole(K=intrinsics)
    
    with torch.no_grad():
        predictions = model.infer(rgb_torch, camera_pinhole)
    
    depth_pinhole = predictions["depth"].squeeze().cpu().numpy()
    print(f"Pinhole depth range: {depth_pinhole.min():.3f}m to {depth_pinhole.max():.3f}m")
    
    # Fisheye camera (example parameters)
    print("\nFisheye camera:")
    # Example fisheye parameters (fx, fy, cx, cy, d1, d2, d3, d4, d5, d6, t1, t2, s1, s2, s3, s4)
    fisheye_params = torch.tensor([1000.0, 1000.0, 640.0, 480.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    camera_fisheye = Fisheye624(params=fisheye_params)
    
    with torch.no_grad():
        predictions = model.infer(rgb_torch, camera_fisheye)
    
    depth_fisheye = predictions["depth"].squeeze().cpu().numpy()
    print(f"Fisheye depth range: {depth_fisheye.min():.3f}m to {depth_fisheye.max():.3f}m")


def main():
    """Run all examples"""
    print("UniDepthV2 Comprehensive Examples")
    print("=" * 50)
    
    try:
        example_1_basic_usage()
        example_2_different_models()
        example_3_with_intrinsics()
        example_4_resolution_levels()
        example_5_interpolation_modes()
        example_6_camera_types()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
