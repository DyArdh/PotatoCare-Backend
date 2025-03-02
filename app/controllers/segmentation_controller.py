from fastapi import UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
from app.model.model import predict, load_unet_model
from app.lib.imagekit import upload_to_imagekit
from app.lib.predict_inform import count_class_pixels, get_predicted_class, get_class_probabilities

# Load model U-Net
model = load_unet_model()

# Define colormap for each class
COLORMAP = {
    0: (0, 0, 0),         # Background: Black
    1: (255, 0, 0),       # Early Blight: Red
    2: (0, 255, 0),       # Healthy: Green
    3: (0, 0, 255)        # Late Blight: Blue
}

def apply_colormap(mask):
    """Apply the colormap to the mask."""
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in COLORMAP.items():
        color_mask[mask == class_id] = color

    return color_mask

async def segment_image(file: UploadFile, return_stream: bool = False):
    # Validasi format file
    if file.content_type != "image/png":
        return {"error": "File format not supported. Please upload a PNG image."}

    # Baca file gambar
    image = Image.open(file.file).convert("RGB")  # Konversi ke RGB
    image_np = np.array(image)

    # Prediksi mask segmentasi
    mask = predict(model, image_np)
    mask_resized = np.argmax(mask[0], axis=-1)  # Pilih kelas dengan probabilitas tertinggi

    # Terapkan colormap pada mask
    mask_colored = apply_colormap(mask_resized)
    mask_image = Image.fromarray(mask_colored)

    # Gabungkan original dengan mask
    blended = Image.blend(image, mask_image.convert("RGB"), alpha=0.5)

    if return_stream:
        # Streaming hasil gambar untuk testing
        output = BytesIO()

        # Pilih gambar yang ingin di-stream
        blended.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(output, media_type="image/png")

    # Upload hasil ke ImageKit dengan nama file dinamis
    original_url = upload_to_imagekit("original", image)
    mask_url = upload_to_imagekit("mask", mask_image)
    blended_url = upload_to_imagekit("blended", blended)
    
    class_pixel_counts = count_class_pixels(mask_resized, exclude_background=True)
    
    total_pixels = mask_resized.size  # Total pixels in the mask
    class_probabilities = get_class_probabilities(class_pixel_counts, total_pixels, exclude_background=True)

    predicted_class = get_predicted_class(class_pixel_counts, exclude_background=True)
    
    print("=================================")
    print(class_probabilities)
    print(predicted_class)
    print(class_pixel_counts)
    

    # Kembalikan hasil URL
    return JSONResponse(
      {
        "images": {
          "original_url": original_url,
          "mask_url": mask_url,
          "blended_url": blended_url
        },
        "predictions": {
          "predicted_class": int(predicted_class),
          # "class_pixel_counts": class_pixel,
          # "class_probabilities": class_probabilities
        }
      },
      status_code=200
    )
