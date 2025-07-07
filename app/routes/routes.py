import os
from fastapi import APIRouter, Depends, File, UploadFile, BackgroundTasks, Body, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import List, Dict
import cuid
from pydantic import BaseModel
import datetime

from app.middlewares.auth import JWTBearerMiddleware
from app.controllers.segmentation_controller import segment_image
from app.controllers.image_management_controller import copy_segmentation_task, process_images_zip

API_VERSION = os.getenv("API_VERSION", "v1")

router = APIRouter(prefix=f"/api/{API_VERSION}")
dataset_prefix = APIRouter(prefix="/dataset")

@router.get("/")
async def root():
    return JSONResponse(
      status_code=200,
      content={"message": "Welcome to PotatoCare API"}
    )
    
@router.get("/protected", dependencies=[Depends(JWTBearerMiddleware())])
async def protected():
    return JSONResponse(
      status_code=200,
      content={"message": "Protect Route"}
    )
    
@router.post("/segmentation")
async def segmentation(image: UploadFile = File(...)):
  return await segment_image(image)

@dataset_prefix.post("/copy-image")
async def copyImage(data: Dict[str, List[str]] = Body(...), background_tasks: BackgroundTasks = None):
    image_ids = data.get("image_ids", [])
    for image_id in image_ids:
        background_tasks.add_task(copy_segmentation_task, image_id)

    return JSONResponse(content={"message": "Proses background dimulai"}, status_code=202)
  
class ImageIds(BaseModel):
    image_ids: list[str]  
@dataset_prefix.post("/process-zip")
async def download_images_zip(image_ids_data: ImageIds, background_tasks: BackgroundTasks):
    """
    Menjadwalkan pembuatan ZIP dari file gambar berdasarkan ID di background, mengembalikan ID file (CUID).
    """
    file_id = cuid.cuid()
    now = datetime.datetime.now()
    date_string = now.strftime("%Y%m%d")
    file_zip_name = f"{date_string}_{file_id}"
    background_tasks.add_task(process_images_zip, image_ids_data.image_ids, file_zip_name)
    return JSONResponse(content={"file_name": file_zip_name, "message": "Pemrosesan gambar sedang berlangsung di background."}, status_code=202)
  
@dataset_prefix.get("/download_zip/{file_name}")
async def download_zip(file_name: str):
    """
    Mengunduh file ZIP berdasarkan nama file (tanpa ekstensi).
    """
    compressed_dir = os.path.join("asset", "compressed")
    full_file_name = f"{file_name}.zip"  # Tambahkan ekstensi .zip
    file_path = os.path.join(compressed_dir, full_file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan")

    return Response(
        open(file_path, "rb").read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={full_file_name}"},
    )
