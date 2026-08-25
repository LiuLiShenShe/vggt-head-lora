#!/usr/bin/env python3
"""
Simple web interface for UniDepthV2
Run this in Docker container for web-based depth estimation
"""

import os
import io
import base64
import numpy as np
import torch
from PIL import Image
import gradio as gr
from unidepth.models import UniDepthV2
from unidepth.utils import colorize


class UniDepthV2Web:
    def __init__(self):
        """Initialize the web interface"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading UniDepthV2 model...")
        self.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        self.model = self.model.to(self.device).eval()
        print("Model loaded successfully!")
        
        # Create input/output directories
        os.makedirs("/app/input", exist_ok=True)
        os.makedirs("/app/output", exist_ok=True)
    
    def process_image(self, image, use_intrinsics=False, intrinsics_file=None):
        """Process image and return depth map"""
        try:
            # Convert image to tensor
            if isinstance(image, str):
                # Handle file path
                rgb = np.array(Image.open(image))
            else:
                # Handle uploaded image
                rgb = np.array(image)
            
            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            
            # Handle intrinsics
            camera = None
            if use_intrinsics and intrinsics_file is not None:
                try:
                    intrinsics = torch.from_numpy(np.load(intrinsics_file.name))
                    from unidepth.utils.camera import Pinhole
                    camera = Pinhole(K=intrinsics)
                    print("Using provided intrinsics")
                except:
                    print("Failed to load intrinsics, using default")
            
            # Run inference
            with torch.no_grad():
                if camera is not None:
                    predictions = self.model.infer(rgb_torch, camera)
                else:
                    predictions = self.model.infer(rgb_torch)
            
            # Extract results
            depth = predictions["depth"].squeeze().cpu().numpy()
            confidence = predictions.get("confidence", None)
            if confidence is not None:
                confidence = confidence.squeeze().cpu().numpy()
            
            # Colorize depth map
            depth_col = colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
            
            # Save outputs
            output_name = f"depth_{np.random.randint(10000)}.png"
            output_path = f"/app/output/{output_name}"
            Image.fromarray(depth_col).save(output_path)
            
            # Convert to base64 for display
            buffered = io.BytesIO()
            Image.fromarray(depth_col).save(buffered, format="PNG")
            depth_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create confidence visualization if available
            confidence_viz = None
            if confidence is not None:
                confidence_viz = Image.fromarray((confidence * 255).astype(np.uint8))
                buffered = io.BytesIO()
                confidence_viz.save(buffered, format="PNG")
                confidence_b64 = base64.b64encode(buffered.getvalue()).decode()
                confidence_viz = f"data:image/png;base64,{confidence_b64}"
            
            return (
                f"data:image/png;base64,{depth_b64}",
                confidence_viz,
                f"Depth range: {depth.min():.3f}m to {depth.max():.3f}m",
                f"Output saved to: {output_path}"
            )
            
        except Exception as e:
            return None, None, f"Error: {str(e)}", "Processing failed"
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="UniDepthV2 Web Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🚀 UniDepthV2 Web Interface")
            gr.Markdown("Upload an image to get metric depth estimation using UniDepthV2")
            
            with gr.Row():
                with gr.Column():
                    # Input section
                    input_image = gr.Image(label="Input Image", type="pil")
                    
                    with gr.Row():
                        use_intrinsics = gr.Checkbox(label="Use Camera Intrinsics", value=False)
                        intrinsics_file = gr.File(label="Intrinsics (.npy)", visible=False)
                    
                    process_btn = gr.Button("🚀 Process Image", variant="primary")
                    
                    # Show/hide intrinsics file based on checkbox
                    use_intrinsics.change(
                        fn=lambda x: gr.File(visible=x),
                        inputs=use_intrinsics,
                        outputs=intrinsics_file
                    )
                
                with gr.Column():
                    # Output section
                    depth_output = gr.Image(label="Depth Map", type="pil")
                    confidence_output = gr.Image(label="Confidence Map", type="pil")
                    depth_info = gr.Textbox(label="Depth Information", interactive=False)
                    status_info = gr.Textbox(label="Status", interactive=False)
            
            # Process button action
            process_btn.click(
                fn=self.process_image,
                inputs=[input_image, use_intrinsics, intrinsics_file],
                outputs=[depth_output, confidence_output, depth_info, status_info]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["assets/demo/rgb.png", False, None],
                ],
                inputs=[input_image, use_intrinsics, intrinsics_file],
                label="Example Images"
            )
        
        return interface


def main():
    """Main function to run the web interface"""
    print("Starting UniDepthV2 Web Interface...")
    
    # Initialize web interface
    web_interface = UniDepthV2Web()
    
    # Create and launch interface
    interface = web_interface.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
