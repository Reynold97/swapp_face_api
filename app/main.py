from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.pipe.components.analyzer import FaceAnalyzer
from src.pipe.components.swapper import FaceSwapper
from src.pipe.components.enhancer import FaceEnhancer
from src.utils import conditional_download, resolve_relative_path, read_image_as_array
from src.globals import execution_providers,similar_face_distance,many_faces

app = FastAPI(title="Image Processing Service")

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

# Assuming FaceSwapper, FaceEnhancer, and FaceAnalyzer are your class names
face_swapper = None
face_enhancer = None
face_analyzer = None

@app.on_event("startup")
async def startup_event():
    #download the models
    try:
        download_directory_path = resolve_relative_path('../models')
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/inswapper_128.onnx"])
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/GFPGANv1.4.pth"])
    except Exception as e:
        print(f"Could not download the base models: {str(e)}")
    
    global face_swapper, face_enhancer, face_analyzer
    face_swapper = FaceSwapper()
    face_enhancer = FaceEnhancer()
    #face_analyzer = FaceAnalyzer()
    
    #global values
    global execution_providers, many_faces, similar_face_distance
    execution_providers = ["CPUExecutionProvider"]
    many_faces = False
    similar_face_distance = 0.85
    
def get_face_swapper():
    return face_swapper

def get_face_enhancer():
    return face_enhancer

#def get_face_analyzer():
#    return face_analyzer

@app.post("/swapp_img/")
async def process_image(source_image: UploadFile = File(...), target_image: UploadFile = File(...), 
                        swapper: FaceSwapper = Depends(get_face_swapper), 
                        enhancer: FaceEnhancer = Depends(get_face_enhancer), 
                        #analyzer: FaceAnalyzer = Depends(get_face_analyzer)
                        ):
    try:
        source_array = await read_image_as_array(source_image)
        target_array = await read_image_as_array(target_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something was wrong with the images")
     
    try:   
        # Now you can use swapper, enhancer, and analyzer directly
        swapper_result = swapper.process_image(source_array, target_array)
        enhancer_result = enhancer.process_image(None, swapper_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something was wrong with the processing")
    
    return enhancer_result