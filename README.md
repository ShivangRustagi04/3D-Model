
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
```

# Requirements.txt

python>=3.8

torch>=1.12.0

numpy>=1.21.0

opencv-python>=4.5.0

rembg>=2.0.0

pyrender>=0.1.45

shap-e@git+https://github.com/openai/shap-e.git

# My thought process

Thought Process
Image Processing:

Started with background removal as it's crucial for good 3D reconstruction

Used RemBG for its simplicity and effectiveness

Planned to use Pixel2Mesh++ for actual conversion but implemented a placeholder for prototype

Text to 3D:

Chose OpenAI's Shap-E as it's currently one of the best open-source text-to-3D models

Implemented the basic pipeline from their GitHub repository

Visualization:

Used Pyrender for simple 3D visualization

Focused on making it work rather than advanced features for this prototype


## Notes

1. For a production system, I would:
   - Implement proper Pixel2Mesh++ integration for image conversion
   - Add more sophisticated preprocessing
   - Implement error handling and input validation
   - Add support for more 3D formats
   - Improve visualization options

2. The current prototype demonstrates:
   - Understanding of the 3D generation pipeline
   - Ability to work with different input modalities
   - Integration of multiple open-source tools
   - Clear documentation and setup instructions

3. To run the text-to-3D part, users will need a GPU with sufficient VRAM. The image-to-3D part can run on CPU.

This implementation balances functionality with the prototype nature of the assignment, showing both current capabilities and understanding of what would be needed for a full implementation.
