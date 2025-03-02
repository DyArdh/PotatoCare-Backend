from fastapi import Request
from fastapi.responses import JSONResponse

async def not_found_handler(req: Request, exc):
  return JSONResponse(
    status_code=404,
    content={"error": "Not Found", "message": "The requested resource was not found on the server."}
  )
  
async def internal_server_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "message": "An unexpected error occurred. Please try again later."},
    )