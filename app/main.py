from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

# Local Import
from app.config import config
from app.middlewares.exceptions import not_found_handler, internal_server_error_handler
from app.routes.routes import router as routes
from app.middlewares.upload_limit import limit_upload_size

# Load Environment Variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Middlewares
app.middleware("http")(limit_upload_size)

# Routes
app.include_router(routes)

# Error Handlers
app.add_exception_handler(404, not_found_handler)
app.add_exception_handler(500, internal_server_error_handler)

if __name__ == "__main__":
  uvicorn.run(
    "app.main:app",
    host=config.SERVER_HOST,
    port=config.SERVER_PORT,
    reload=config.DEBUG
  )