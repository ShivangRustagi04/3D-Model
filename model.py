import os
import argparse
import numpy as np
import torch
from PIL import Image
import cv2
from rembg import remove
import pyrender
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import decode_latent_mesh

class ImageTo3DConverter:
    def __init__(self):
        # Initialize Pixel2Mesh++ would go here
        # For prototype, we'll use a simplified approach
        pass
    
    def remove_background(self, image_path):
        """Remove background from image using rembg"""
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        return output_image
    
    def preprocess_image(self, image):
        """Preprocess image for 3D conversion"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) > 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Normalize and resize
        img_array = cv2.resize(img_array, (256, 256))
        img_array = img_array / 255.0
        
        return img_array
    
    def generate_3d_from_image(self, image_path, output_path):
        """Generate 3D model from image"""
        # Remove background
        segmented_img = self.remove_background(image_path)
        
        # Preprocess
        processed_img = self.preprocess_image(segmented_img)
        
        # In a full implementation, we would feed this to Pixel2Mesh++
        # For prototype, we'll create a simple placeholder 3D shape
        
        # Generate simple 3D cube (placeholder for actual model)
        vertices = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [1, 2, 6], [1, 6, 5],  # right
            [0, 3, 7], [0, 7, 4]   # left
        ], dtype=np.uint32)
        
        # Save as OBJ
        self.save_obj(vertices, faces, output_path)
        
        return vertices, faces
    
    def save_obj(self, vertices, faces, filepath):
        """Save 3D model as OBJ file"""
        with open(filepath, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def visualize_3d(self, vertices, faces):
        """Visualize 3D model using pyrender"""
        mesh = pyrender.Mesh.from_points(vertices)
        scene = pyrender.Scene()
        scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)

class TextTo3DConverter:
    def __init__(self):
        # Load Shap-E models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.xm = load_model('transmitter', device=self.device)
        self.model = load_model('text300M', device=self.device)
    
    def generate_3d_from_text(self, prompt, output_path):
        """Generate 3D model from text prompt using Shap-E"""
        # Generate latent
        batch_size = 1
        guidance_scale = 15.0
        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.xm,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        
        # Decode latent to mesh
        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()
            with open(output_path, 'w') as f:
                t.write_obj(f)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert image or text to 3D model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--text', type=str, help='Text prompt for 3D generation')
    parser.add_argument('--output', type=str, default='output.obj', help='Output file path')
    args = parser.parse_args()
    
    if args.image:
        converter = ImageTo3DConverter()
        vertices, faces = converter.generate_3d_from_image(args.image, args.output)
        print(f"3D model saved to {args.output}")
        converter.visualize_3d(vertices, faces)
    elif args.text:
        converter = TextTo3DConverter()
        output_path = converter.generate_3d_from_text(args.text, args.output)
        print(f"3D model saved to {output_path}")
    else:
        print("Please provide either --image or --text input")

if __name__ == "__main__":
    main()
