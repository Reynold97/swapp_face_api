import threading
from typing import Any, Optional, List
import numpy
import os

import insightface
from insightface.app.common import Face

from src.typing import Frame
import src.globals
from src.pipe.components.processor import Processor

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from dotenv import load_dotenv

from src.utils import conditional_download, resolve_relative_path, read_image_as_array, suggest_execution_providers


class FaceAnalyzer(Processor):      
    def __init__(self, providers: List[str]):
        super().__init__('FACE_ANALYSER', '', providers)  # model path is blank because is downloaded from insightface, not local.
        self.load_model() # Ensures model is loaded during instantiation
       
    def load_model(self) -> Any:
        with threading.Lock():
            if self.model is None:
                self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.model.prepare(ctx_id=0)
        return self.model

    def get_one_face(self, frame: Frame, position: int = 0) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            try:
                return many_faces[position]
            except IndexError:
                return many_faces[-1]
        return None

face_analyzer = None

def get_face_analyzer():
    return face_analyzer


app = FastAPI(title="Image Processing Service")
load_dotenv(".env")

@app.on_event("startup")
async def startup_event():
    #download the models
    try:
        download_directory_path = resolve_relative_path('../models')
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/inswapper_128.onnx"])
        conditional_download(download_directory_path, ["https://huggingface.co/Reynold97/swapp_models/resolve/main/GFPGANv1.4.pth"])
    except Exception as e:
        print(f"Could not download the base models: {str(e)}")
    
    providers_str = os.getenv('PROVIDERS', 'CPUExecutionProvider')  # Defaulting to CPUExecutionProvider if not set
    execution_providers = providers_str.split(',')
    global face_analyzer
    face_analyzer = FaceAnalyzer(execution_providers)
    
@app.get("/")
async def checkhealth(self):
    return Response(status_code=200)
    
@app.post("/")
async def process_image(model: UploadFile = File(...), 
                        face: UploadFile = File(...),
                        watermark: Optional[bool] = Form(None), 
                        vignette: Optional[bool] = Form(None),  
                        #swapper: FaceSwapper = Depends(get_face_swapper), 
                        #enhancer: FaceEnhancer = Depends(get_face_enhancer), 
                        analyzer: FaceAnalyzer = Depends(get_face_analyzer),
                        #image_pipeline = Depends(get_image_pipeline),
                        ):
    try:
        source_array = await read_image_as_array(face)
        target_array = await read_image_as_array(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something was wrong with the images")
    
    try:
        source_face = analyzer.get_one_face(source_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analyzer don't work")  
    
    return Response(source_face)
    
    
"""
Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider'], with options: {'CPUExecutionProvider': {}, 
'CUDAExecutionProvider': {'tunable_op_max_tuning_duration_ms': '0', 'enable_skip_layer_norm_strict_mode': '0', 
'tunable_op_tuning_enable': '0', 'device_id': '0', 'has_user_compute_stream': '0', 
'gpu_mem_limit': '18446744073709551615', 'gpu_external_alloc': '0', 'gpu_external_free': '0', 
'gpu_external_empty_cache': '0', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'cudnn_conv1d_pad_to_nc1d': '0', 
'arena_extend_strategy': 'kNextPowerOfTwo', 'do_copy_in_default_stream': '1', 'enable_cuda_graph': '0', 
'cudnn_conv_use_max_workspace': '1', 'tunable_op_enable': '0'}}
"""