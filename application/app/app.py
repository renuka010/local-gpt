#main.py

# Imports
import os

# FastAPI import
from fastapi import FastAPI, Response, status

# Package imports
from logs import LoggerConfig
from logs import logger
from schemas import ResponseModel
from settings import DEBUG_MODE

from routes import router as service_routes


# from environment
debug_mode = DEBUG_MODE


# init logs
logging_config = LoggerConfig()
    
# Initialize application
app = FastAPI(title="Python API's NSDC",
            debug=debug_mode)

@app.on_event("startup")
def startup_event():
    logging_config.setup_logging()


# Add routes
app.include_router(service_routes, prefix="/langchain", tags={"Local-llm-gpt"})



@app.get("/", status_code=status.HTTP_200_OK)
def fastapi(response: Response):
    response.status_code = status.HTTP_200_OK
    info = {
        "status": response.status_code,
        "message" :"Mistral-local-gpt",
        "language": "Python:3.8.10",
        "framework": "FastAPI:0.85.0"
    }
    logger.info(f"Application initialized: {info}")
    return ResponseModel(
            status="Ok",
            code="200",
            message="Application Service Running",
            result = info
       )