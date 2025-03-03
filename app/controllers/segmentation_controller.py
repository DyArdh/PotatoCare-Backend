import os
import cv2
import numpy as np
import cuid
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from app.model.model import predict, load_unet_model
from app.lib.predict_inform import count_pixels_per_class, calculate_class_percentages

model = load_unet_model()

COLORMAP = {
    0: (0, 0, 0),    
    1: (255, 0, 0),  
    2: (0, 255, 0),  
    3: (0, 0, 255)      
}

BASE_DIR = "asset/segmentation"
ORIGINAL_DIR = os.path.join(BASE_DIR, "original")
MASK_NPY_DIR = os.path.join(BASE_DIR, "mask_npy")  
MASK_PNG_DIR = os.path.join(BASE_DIR, "mask_png")  
OVERLAY_DIR = os.path.join(BASE_DIR, "overlay")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(MASK_NPY_DIR, exist_ok=True)
os.makedirs(MASK_PNG_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)

def apply_colormap(mask):
    """Apply the colormap to the mask."""
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in COLORMAP.items():
        color_mask[mask == class_id] = color
    return color_mask

async def segment_image(file: UploadFile):
    if file.content_type != "image/png":
        return {"error": "File format not supported. Please upload a PNG image."}

    file_id = cuid.cuid()

    original_path = os.path.join(ORIGINAL_DIR, f"{file_id}.png")
    mask_npy_path = os.path.join(MASK_NPY_DIR, f"{file_id}.npy")
    mask_png_path = os.path.join(MASK_PNG_DIR, f"{file_id}.png")
    blended_path = os.path.join(OVERLAY_DIR, f"{file_id}.png")

    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)

    mask = predict(model, image_np)
    mask_resized = np.argmax(mask[0], axis=-1) 

    np.save(mask_npy_path, mask_resized)

    mask_colored = apply_colormap(mask_resized)

    cv2.imwrite(original_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)) 
    cv2.imwrite(mask_png_path, mask_colored) 
    blended = cv2.addWeighted(image_np, 0.5, mask_colored, 0.5, 0)  
    cv2.imwrite(blended_path, blended)  

    class_pixel_counts = count_pixels_per_class(mask_resized, exclude_background=True)
    class_percentages = calculate_class_percentages(class_pixel_counts)

    return JSONResponse(
        {
            "id": file_id,
            "images": {
                "original_url": f"/asset/segmentation/original/{file_id}.png",
                "mask_npy_url": f"/asset/segmentation/mask_npy/{file_id}.npy",
                "mask_png_url": f"/asset/segmentation/mask_png/{file_id}.png",
                "blended_url": f"/asset/segmentation/overlay/{file_id}.png"
            },
            "predictions": {
                "total_pixels": class_pixel_counts,
                "class_percentages": class_percentages
            }
        },
        status_code=200
    )