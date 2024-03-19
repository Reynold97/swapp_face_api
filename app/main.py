from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.pipe.components.analyzer import FaceAnalyzer
from src.pipe.components.swapper import FaceSwapper
from src.pipe.components.enhancer import FaceEnhancer
from src.pipe.pipeline import ImagePipeline
from src.utils import conditional_download, resolve_relative_path, read_image_as_array, suggest_execution_providers
from src.globals import similar_face_distance,many_faces, reference_face_position
from typing import Optional
from dotenv import load_dotenv
import os
from PIL import Image
import io

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

# Assuming FaceSwapper, FaceEnhancer, FaceAnalyzer and Image_Pipeline are the class names
#face_swapper = None
#face_enhancer = None
#face_analyzer = None
image_pipeline = None

@app.on_event("startup")
async def startup_event():
    #download the models
    try:
        download_directory_path = resolve_relative_path('../models')
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/inswapper_128.onnx"])
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/GFPGANv1.4.pth"])
    except Exception as e:
        print(f"Could not download the base models: {str(e)}")
    
    #global values
    #global many_faces, similar_face_distance, reference_face_position
    #many_faces = False
    #similar_face_distance = 0.85
    #reference_face_position = 0
    
    providers_str = os.getenv('PROVIDERS', 'CPUExecutionProvider')  # Defaulting to CPUExecutionProvider if not set
    execution_providers = providers_str.split(',')
    #global face_swapper, face_enhancer, face_analyzer
    #face_swapper = FaceSwapper(execution_providers)
    #face_enhancer = FaceEnhancer(execution_providers)
    #face_analyzer = FaceAnalyzer()
    global image_pipeline
    image_pipeline = ImagePipeline(execution_providers)
    
#def get_face_swapper():
#    return face_swapper

#def get_face_enhancer():
#    return face_enhancer

#def get_face_analyzer():
#    return face_analyzer

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
                        #swapper: FaceSwapper = Depends(get_face_swapper), 
                        #enhancer: FaceEnhancer = Depends(get_face_enhancer), 
                        #analyzer: FaceAnalyzer = Depends(get_face_analyzer),
                        image_pipeline = Depends(get_image_pipeline),
                        ):
    try:
        source_array = await read_image_as_array(face)
        target_array = await read_image_as_array(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something was wrong with the images")
    
    #many_faces = False
    #similar_face_distance = 0.85
    #reference_face_position = 0
     
    try:   
        # Old version use swapper, enhancer, and analyzer directly
        #swapper_result = swapper.process_image(source_array, target_array)
        #enhancer_result = enhancer.process_image(None, swapper_result)
        pipeline_result = image_pipeline.process_1_image_default_face(source_array, target_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something was wrong with the processing")
    
    # Convert from BGR to RGB
    result = pipeline_result[:, :, ::-1]  

    # Convertir el array en una imagen
    image = Image.fromarray(result)
        
    # Convert PIL Image to a byte stream (in memory)
    byte_io = io.BytesIO()
    image.save(byte_io, format="PNG")
    byte_io.seek(0)  # Go back to the start of the bytes stream

    # Return image as a stream
    return StreamingResponse(byte_io, media_type="image/png")
    