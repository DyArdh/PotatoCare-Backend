import os
import shutil
import zipfile
from io import BytesIO
import aiofiles
import asyncio

BASE_DIR = "asset/segmentation"
ORIGINAL_DIR = os.path.join(BASE_DIR, "images")
MASK_NPY_DIR = os.path.join(BASE_DIR, "masks")

# Direktori dataset
DATASET_DIR = "asset/dataset"
DATASET_ORIGINAL_DIR = os.path.join(DATASET_DIR, "images")
DATASET_MASK_NPY_DIR = os.path.join(DATASET_DIR, "masks")

# Pastikan direktori dataset ada
os.makedirs(DATASET_ORIGINAL_DIR, exist_ok=True)
os.makedirs(DATASET_MASK_NPY_DIR, exist_ok=True)

def copy_segmentation_task(image_id: str):
    original_file = os.path.join(ORIGINAL_DIR, f"{image_id}.png")
    mask_npy_file = os.path.join(MASK_NPY_DIR, f"{image_id}.npy")

    if not os.path.exists(original_file):
        print(f"File original tidak ditemukan untuk image_id: {image_id}")
        return
    if not os.path.exists(mask_npy_file):
        print(f"File mask .npy tidak ditemukan untuk image_id: {image_id}")
        return

    try:
        shutil.copy(original_file, os.path.join(DATASET_ORIGINAL_DIR, f"{image_id}.png"))
        shutil.copy(mask_npy_file, os.path.join(DATASET_MASK_NPY_DIR, f"{image_id}.npy"))
        
    except Exception as e:
        print(f"Gagal mengcopy file untuk image_id: {image_id}, error: {str(e)}")
        

async def process_images_zip(image_ids: list[str], file_zip_name: str):
    """
    Memproses gambar dan mask berdasarkan ID dan membuat file ZIP di background, menyimpan di asset/compressed.
    """
    zip_buffer = BytesIO()
    images_dir = os.path.join("asset", "dataset", "images")
    masks_dir = os.path.join("asset", "dataset", "masks")
    compressed_dir = os.path.join("asset", "compressed")

    # Pastikan direktori hasil ada
    os.makedirs(compressed_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            tasks = []
            for image_id in image_ids:
                image_found = False
                mask_found = False

                for filename in os.listdir(images_dir):
                    if filename.startswith(image_id):
                        image_path = os.path.join(images_dir, filename)
                        if os.path.isfile(image_path):
                            tasks.append(process_image(image_path, filename, zip_file))
                            image_found = True
                            break

                for filename in os.listdir(masks_dir):
                    if filename.startswith(image_id):
                        mask_path = os.path.join(masks_dir, filename)
                        if os.path.isfile(mask_path):
                            tasks.append(process_image(mask_path, filename, zip_file))
                            mask_found = True
                            break

                if not image_found:
                    print(f"Gambar dengan ID {image_id} tidak ditemukan")
                if not mask_found:
                    print(f"Mask dengan ID {image_id} tidak ditemukan")

            await asyncio.gather(*tasks)

        # Simpan file zip di direktori asset/compressed dengan format nama YYYYMMDD_file_id.zip

        file_name = f"{file_zip_name}.zip"
        file_path = os.path.join(compressed_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(zip_buffer.getvalue())

    except Exception as e:
        print(f"Error saat memproses gambar dan mask: {e}")

async def process_image(file_path: str, filename: str, zip_file: zipfile.ZipFile):
    """
    Memproses satu file (gambar atau mask) secara asynchronous.
    """
    try:
        async with aiofiles.open(file_path, "rb") as f:
            file_content = await f.read()
            zip_file.writestr(filename, file_content)
    except Exception as e:
        print(f"Error saat menambahkan {filename} ke zip: {e}")