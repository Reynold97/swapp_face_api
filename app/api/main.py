from typing import List, Optional, Union, Any
import yaml
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import cv2
import numpy as np
from io import BytesIO

from app.api.schemas import (
    SwapUrlRequest, SwapImgRequest, ProcessingOptions, SuccessResponse, 
    PartialSuccessResponse, ErrorResponse, FailedModel
)
from app.pipe.components.analyzer import FaceAnalyzer
from app.pipe.components.enhancer import FaceEnhancer
from app.pipe.components.swapper import FaceSwapper
from app.pipe.components.enhancer_codeformer import CodeFormerEnhancer
from app.pipe.swap_processor import SwapProcessor, NoSourceFaceError, NoTargetFaceError
from app.services.gcp_bucket_manager import GCPImageManager
from app.utils.utils import conditional_download

@asynccontextmanager
async def lifespan(app: FastAPI): 
    """
    Lifecycle manager for the FastAPI application.
    Downloads required models during startup.
    """
    # Load the pipeline models
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/inswapper_128_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/arcface_w600k_r50_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/retinaface_10g_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/gfpgan_1.4.onnx')
    
    # CodeFormer Models
    conditional_download(
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'models/weights/facelib'
    )
    conditional_download(
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'models/weights/facelib'
    )
    conditional_download(
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'models/weights/CodeFormer'
    )
    conditional_download(
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
        'models/weights/realesrgan'
    )
    
    yield
    
# Create FastAPI app with metadata
app = FastAPI(
    title="Face Swap API",
    description="API for face swapping with multiple modes and enhancement options",
    version="2.0.0",
    lifespan=lifespan
)

@serve.deployment()
@serve.ingress(app)
class APIIngress:
    """
    API Ingress for face swap operations using Ray Serve.
    
    This class provides endpoints for face swapping operations with different modes:
    - one_to_one: Standard single face swap (default)
    - one_to_many: Apply one source face to all target faces
    - sorted: Apply faces in spatial order (left-to-right or right-to-left)
    - similarity: Match source faces to target faces by similarity
    
    It supports both URL-based operations (using GCP storage) and direct image upload.
    """
    def __init__(self, 
                swap_processor: DeploymentHandle,
                img_manager: DeploymentHandle):
        """
        Initialize the API Ingress with handles to required components.
        
        Args:
            swap_processor: Handle to the SwapProcessor deployment
            img_manager: Handle to the GCPImageManager deployment
        """
        self.swap_processor = swap_processor
        self.img_manager = img_manager

    @app.post("/swap_url", 
              response_model=Union[SuccessResponse, PartialSuccessResponse],
              responses={
                  200: {"model": SuccessResponse, "description": "Successful operation"},
                  206: {"model": PartialSuccessResponse, "description": "Partially successful operation"},
                  400: {"model": ErrorResponse, "description": "Bad request"},
                  500: {"model": ErrorResponse, "description": "Server error"}
              },
              tags=["Face Swap"],
              summary="Swap faces using images from GCP storage")
    async def swap_url(self, request: SwapUrlRequest):
        """
        Perform face swapping using images stored in GCP bucket.
        
        - **model_filenames**: List of target image filenames in the GCP bucket
        - **face_filename**: Source face image filename in the GCP bucket
        - **options**: Processing options including:
            - **mode**: Swap mode (one_to_one, one_to_many, sorted, similarity)
            - **direction**: Direction for sorted mode (left_to_right, right_to_left)
            - **use_codeformer**: Whether to use CodeFormer for enhancement
            - **codeformer_fidelity**: Balance between quality and fidelity (0-1)
            - **background_enhance**: Whether to enhance the background
            - **face_upsample**: Whether to upsample the faces
            - **upscale**: Upscale factor for enhancement
        
        Returns:
            JSON response with URLs of processed images or error details
        """
        # Extract request data
        face_filename = request.face_filename
        model_filenames = request.model_filenames
        options = request.options
            
        # Download source image
        source_img = await self.img_manager.download_image.remote(face_filename)
        
        if source_img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Bad Request", "message": f"Failed to download image: {face_filename}"}
            )
        
        urls = []
        failed_models = []
        
        # Process each target image
        for model_filename in model_filenames:
            try:
                # Download target image
                target_img = await self.img_manager.download_image.remote(model_filename)
                
                if target_img is None:
                    failed_models.append(FailedModel(
                        filename=model_filename,
                        error="Failed to download image"
                    ))
                    continue
                
                # Process the swap using the SwapProcessor
                try:
                    result_img = await self.swap_processor.process_swap.remote(
                        mode=options.mode,
                        source_frame=source_img,
                        target_frame=target_img,
                        direction=options.direction,
                        enhance=True,
                        use_codeformer=options.use_codeformer,
                        codeformer_fidelity=options.codeformer_fidelity,
                        background_enhance=options.background_enhance,
                        face_upsample=options.face_upsample,
                        upscale=options.upscale
                    )
                except NoSourceFaceError as e:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={"error": "Bad Request", "message": str(e)}
                    )
                except NoTargetFaceError as e:
                    # For target face errors, we continue with the next model
                    failed_models.append(FailedModel(
                        filename=model_filename,
                        error=str(e)
                    ))
                    continue
                
                # Upload the result
                url = await self.img_manager.upload_image.remote(result_img)
                if url:
                    urls.append(url)
                else:
                    failed_models.append(FailedModel(
                        filename=model_filename,
                        error="Failed to upload result image"
                    ))
                
            except HTTPException:
                raise
            except Exception as e:
                failed_models.append(FailedModel(
                    filename=model_filename,
                    error=str(e)
                ))
        
        # Determine response status code and content
        if len(failed_models) > 0 and len(urls) > 0:
            # Partial success
            return JSONResponse(
                status_code=status.HTTP_206_PARTIAL_CONTENT,
                content=jsonable_encoder(PartialSuccessResponse(urls=urls, failed=failed_models))
            )
        elif len(failed_models) > 0 and len(urls) == 0:
            # All failed
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Processing Failed", "details": [jsonable_encoder(model) for model in failed_models]}
            )
        else:
            # All succeeded
            return SuccessResponse(urls=urls)

    @app.post("/swap_img",
              responses={
                  200: {"description": "Successful operation", "content": {"image/png": {}}},
                  400: {"model": ErrorResponse, "description": "Bad request"},
                  500: {"model": ErrorResponse, "description": "Server error"}
              },
              tags=["Face Swap"],
              summary="Swap faces using uploaded images")
    async def swap_img(self, 
                      model: UploadFile = File(..., description="Target image file"),
                      face: UploadFile = File(..., description="Source face image file"),
                      options: ProcessingOptions = Depends()):
        """
        Perform face swapping using directly uploaded images.
        
        - **model**: Target image file upload
        - **face**: Source face image file upload
        - **options**: Processing options including:
            - **mode**: Swap mode (one_to_one, one_to_many, sorted, similarity)
            - **direction**: Direction for sorted mode (left_to_right, right_to_left)
            - **use_codeformer**: Whether to use CodeFormer for enhancement
            - **codeformer_fidelity**: Balance between quality and fidelity (0-1)
            - **background_enhance**: Whether to enhance the background
            - **face_upsample**: Whether to upsample the faces
            - **upscale**: Upscale factor for enhancement
        
        Returns:
            The processed image
        """
        try:
            # Load source and target images
            source_img = self.__load_image(await face.read())
            target_img = self.__load_image(await model.read())
            
            if source_img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": "Invalid source image format"}
                )
                
            if target_img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": "Invalid target image format"}
                )
            
            # Process the swap using the SwapProcessor
            try:
                result_img = await self.swap_processor.process_swap.remote(
                    mode=options.mode,
                    source_frame=source_img,
                    target_frame=target_img,
                    direction=options.direction,
                    enhance=True,
                    use_codeformer=options.use_codeformer,
                    codeformer_fidelity=options.codeformer_fidelity,
                    background_enhance=options.background_enhance,
                    face_upsample=options.face_upsample,
                    upscale=options.upscale
                )
            except NoSourceFaceError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": str(e)}
                )
            except NoTargetFaceError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": str(e)}
                )
            
            # Convert result to bytes for response
            img_bytes = self.__result_image_bytes(result_img)
            
            return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Processing Failed", "message": str(e)}
            )

    def __load_image(self, img_bytes: bytes) -> np.ndarray:
        """
        Load an image from bytes.
        
        Args:
            img_bytes: Image data as bytes
            
        Returns:
            NumPy array containing the image
        """
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def __result_image_bytes(self, image: np.ndarray) -> bytes:
        """
        Convert an image to bytes.
        
        Args:
            image: Image as NumPy array
            
        Returns:
            Image encoded as bytes
        """
        _, img_encoded = cv2.imencode('.png', image)
        return img_encoded.tobytes()


# Load the GCP Bucket config
# Load configuration from YAML
with open('app/configs/serve_config_gpu.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract GCPImageManager init_args from config
gcp_image_manager_config = None
for app in config['applications']:
    if app['name'] == 'SwapFaceAPI':
        for deployment in app['deployments']:
            if deployment['name'] == 'GCPImageManager':
                gcp_image_manager_config = deployment['init_args']

if gcp_image_manager_config is None:
    raise ValueError("GCPImageManager configuration not found in config.yaml")

# Bind actors with configuration for SwapProcessor
analyzer = FaceAnalyzer.bind()
swapper = FaceSwapper.bind()
enhancer = FaceEnhancer.bind()
codeformer = CodeFormerEnhancer.bind()
img_manager = GCPImageManager.bind(*gcp_image_manager_config)

# Create the SwapProcessor
swap_processor = SwapProcessor.bind(analyzer, swapper, enhancer, codeformer)

# Simplified binding for API Ingress - only needs SwapProcessor and GCPImageManager
app = APIIngress.bind(swap_processor, img_manager)