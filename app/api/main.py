from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.pipe.components.analyzer import FaceAnalyzer
from app.pipe.components.swapper import FaceSwapper
from app.pipe.components.enhancer import FaceEnhancer
from app.pipe.pipeline import ImagePipeline
from app.utils.utils import conditional_download, resolve_relative_path, read_image_as_array, suggest_execution_providers
from typing import Optional
from dotenv import load_dotenv
import os
from PIL import Image
import io
from io import BytesIO
import ray
from ray import serve
from pyngrok import ngrok
import timeit
import cv2
import asyncio
import numpy as np

app = FastAPI(title="Image Processing Service")
load_dotenv(".env")

# Allowed origins for CORS (Cross-Origin Resource Sharing)
allowed_origins = [
    "*",  # Allows all origins
    # Uncomment the line below to restrict to a specific origin
    # "https://api.storyface.ai"
]

# Apply CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)   

image_pipeline = None

@app.on_event("startup")
async def startup_event():
    #download the models
    try:
        download_directory_path = resolve_relative_path('../models')
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/inswapper_128.onnx"])
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/GFPGANv1.4.pth"])
    except Exception as e:
        print(f"Can't download the base models: {str(e)}")
        raise Exception(e)
    
    providers_str = os.getenv('PROVIDERS', 'CPUExecutionProvider')  # Defaulting to CPUExecutionProvider if not set
    execution_providers = providers_str.split(',')
    
    try:
        global image_pipeline
        image_pipeline = ImagePipeline(execution_providers)
    except Exception as e:
        print(f"Can't initialize the pipeline: {str(e)}")
        raise Exception(e)
    
    #public_url = ngrok.connect("8000").public_url
    #print(f"ngrok tunnel \"{public_url}\" -> \"http://localhost:8000\"")
    
def get_image_pipeline():
    return image_pipeline

@app.get("/")
async def checkhealth():
    return Response(status_code=200)

@app.post("/swapp_face/")
async def process_image(model: UploadFile = File(...), 
                        face: UploadFile = File(...),
                        watermark: Optional[bool] = Form(None), 
                        vignette: Optional[bool] = Form(None),  
                        image_pipeline = Depends(get_image_pipeline),
                        ):
    try:
        source_array = await read_image_as_array(face)
        target_array = await read_image_as_array(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something was wrong with the input images: {str(e)}")
    
    start_time = timeit.default_timer() 
    try:         
        pipeline_result = image_pipeline.process_1_image_default_face(source_array, target_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something was wrong with pipeline inference: {str(e)}")
    end_time = timeit.default_timer()
    execution_time = end_time - start_time 
    print(f"Pipeline Inference: {execution_time} s")
    
    # Convert from BGR to RGB
    #result = pipeline_result[:, :, ::-1]  

    # Convertir el array en una imagen
    #image = Image.fromarray(result)
        
    # Convert PIL Image to a byte stream (in memory)
    #byte_io = io.BytesIO()
    #image.save(byte_io, format="PNG")
    #byte_io.seek(0)  # Go back to the start of the bytes stream
    
    # Return image as a stream
    #return StreamingResponse(byte_io, media_type="image/png")    
         
    # Direct encoding to PNG using OpenCV
    success, encoded_image = cv2.imencode(".png", pipeline_result)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode the processed image")        
    
    #return StreamingResponse(BytesIO(encoded_image), media_type="image/png")
    # Convert encoded image to bytes and return it as a PNG response
    encoded_image_bytes = encoded_image.tobytes()
    return Response(content=encoded_image_bytes, media_type="image/png")
   
    
@serve.deployment()
@serve.ingress(app)
class FastAPIWrapper:
    pass

ray_app = FastAPIWrapper.bind()
