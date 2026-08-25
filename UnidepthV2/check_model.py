#!/usr/bin/env python3
"""
Check if UniDepthV2 model is properly loaded
"""

import torch
from unidepth.models import UniDepthV2


def check_model():
    """Check if UniDepthV2 model can be loaded"""
    
    print("=== UniDepthV2 Model Check ===")
    
    try:
        # Try to load the model
        print("1. Attempting to load UniDepthV2 model...")
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        print("✓ Model loaded successfully!")
        
        # Check model type
        print(f"2. Model type: {type(model)}")
        print(f"3. Model class: {model.__class__.__name__}")
        
        # Check if model has required methods
        print("4. Checking model methods...")
        if hasattr(model, 'infer'):
            print("✓ Model has 'infer' method")
        else:
            print("✗ Model missing 'infer' method")
            
        if hasattr(model, 'forward'):
            print("✓ Model has 'forward' method")
        else:
            print("✗ Model missing 'forward' method")
        
        # Check device placement
        print("5. Checking device placement...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Available device: {device}")
        
        model = model.to(device)
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Check model parameters
        print("6. Checking model parameters...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Check model state
        print("7. Setting model to evaluation mode...")
        model.eval()
        print("✓ Model set to evaluation mode")
        
        print("\n=== Model Check Complete ===")
        print("✓ UniDepthV2 model is properly loaded and ready to use!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_model()
    if success:
        print("\nModel is ready for inference!")
    else:
        print("\nModel failed to load properly.")
