from fastapi import Request
from fastapi.responses import JSONResponse

MAX_UPLOAD_SIZE = 500 * 1024

async def limit_upload_size(request: Request, call_next):
  content_length = request.headers.get("Content-Length")
  if content_length and int(content_length) > MAX_UPLOAD_SIZE:
    return JSONResponse(
      {"message": "File size exceeds the maximum limit of 500 KB."},
      status_code=413
    )
    
  return await call_next(request)