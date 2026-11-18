from ultralytics import YOLO
#local import
import sys
import os

# Adds the root project folder (FRC-YOLO) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Now your existing imports will work perfectly
from src.config import DEVICE
# You MUST put the training code inside this block on Windows
if __name__ == '__main__':
    
    # 1. Load the official YOLO11 Medium pretrained model
    model = YOLO("C:/Users/alibu/Desktop/FRC-YOLO/runs/detect/frc_scouting/custom_annotated_augmented_v1/weights/last.pt")

    # 2. Train on your custom dataset
    results = model.train(
        data="datasets/custom_annotated_v1/data.yaml", 
        epochs=100,
        imgsz=1024,      
        batch=-1,
        workers=8,
        exist_ok= True,       
        resume = True,  
        device=DEVICE,        
        project="frc_scouting",  
        name="custom_annotated_augmented_v1",
        # FRC-Specific Augmentations
        hsv_h=0.015,      # Slight hue shift (keep low so Red/Blue bumpers don't flip color)
        hsv_s=0.7,        # High saturation shift (simulates intense stadium LEDs)
        hsv_v=0.4,        # High value/brightness shift (simulates shadows and glare)
        degrees=10.0,     # Slight rotation (mimics camera sway or a tipping robot)
        translate=0.1,    # Shifting image position up/down/left/right
        scale=0.5,        # Random zooming in/out
        shear=2.0,        # Perspective skewing
        perspective=0.0,  # 3D perspective warp (keep low/off unless camera angle changes drastically)
        flipud=0.0,       # Upside down flip (keep 0.0 for FRC, robots don't drive upside down)
        fliplr=0.5,       # Left/Right flip (50% probability to encounter mirroring)
        mosaic=1.0,       # Stitches 4 images together (forces model to learn partial occlusion)
        erasing=0.4,  # Probability of applying random erasing to an image (0.0 to 1.0)
        mixup=0.1         # Overlays images transparently
    )