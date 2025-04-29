# 3D-Model

# Photo/Text to 3D Model Converter

This prototype converts either images or text prompts into simple 3D models in OBJ format.

## Features
- Image to 3D conversion with background removal
- Text to 3D conversion using Shap-E
- Basic 3D visualization
- Output in OBJ format

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

#requirements.txt

python>=3.8
torch>=1.12.0
numpy>=1.21.0
opencv-python>=4.5.0
rembg>=2.0.0
pyrender>=0.1.45
shap-e@git+https://github.com/openai/shap-e.git
