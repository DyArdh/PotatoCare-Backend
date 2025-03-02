import os
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from app.middlewares.auth import JWTBearerMiddleware
from app.controllers.segmentation_controller import segment_image

API_VERSION = os.getenv("API_VERSION", "v1")

router = APIRouter(prefix=f"/api/{API_VERSION}")

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
  return await segment_image(image, return_stream=False)