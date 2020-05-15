# Setup/Import ----------------------------------------------------------------

# App
import os
import json
import utils
import time

# Import Routes
from routers import process_notebook

# ML Flow
import mlflow
from mlflow.tracking import MlflowClient

# Fast API
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.responses import HTMLResponse, JSONResponse
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Instantiate Fast API --------------------------------------------------------
app = FastAPI(    
    title="Sentiment Analyzer",
    description="Proof of Concept for REST-Based ML",
    version="0.0.2",
)

# Error Handling for App Failures
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):

    sc = status.HTTP_422_UNPROCESSABLE_ENTITY
    return  JSONResponse(
        status_code=sc,
        content=jsonable_encoder({
            "error": {
                "validation_errors": exc.errors(),
                "code": sc
            },
            "timestamp": time.ctime(),
            "reference": "<>"
        })
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    
    if exc.status_code == 404: sc = status.HTTP_404_NOT_FOUND
    elif exc.status_code == 422: sc = status.HTTP_422_UNPROCESSABLE_ENTITY
    else: sc = 500
    
    return JSONResponse(
        status_code=sc,
        content=jsonable_encoder({
            "error": {
                "message": exc.detail,
                "code": sc
            },
            "timestamp": time.ctime(),
            "reference": "<>"
        })
    )

# ROUTERS ---------------------------------------------------------------------
app.include_router(
    process_notebook.router, 
    prefix="/api/v1/sentiment_analysis"
)




