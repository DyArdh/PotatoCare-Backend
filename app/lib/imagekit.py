from imagekitio import ImageKit
from io import BytesIO
from PIL import Image
from datetime import datetime
import base64
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions

from app.config import config

imagekit = ImageKit(
  private_key=config.IMAGEKIT_PRIVATE_KEY,
  public_key=config.IMAGEKIT_PUBLIC_KEY,
  url_endpoint=config.IMAGEKIT_URL_ENDPOINT
)

def upload_to_imagekit(prefix, image: Image.Image):
    # Buat nama file berbasis tanggal dan waktu
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)  # Pastikan pointer di awal buffer
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Verifikasi ukuran buffer
    if buffer.getbuffer().nbytes == 0:
        raise ValueError("Buffer is empty; image data might not be written correctly.")
      
    options = UploadFileRequestOptions(
      use_unique_file_name=True,
      folder=f"/PotatoCare/segmentations/{prefix}/"
    )

    # Unggah ke ImageKit
    upload_response = imagekit.upload_file(
        file=image_base64,
        file_name=filename,
        options=options
    )
    
    return {
      "file_id": upload_response.file_id,
      "file_name": upload_response.name,
      "url": upload_response.url
    }