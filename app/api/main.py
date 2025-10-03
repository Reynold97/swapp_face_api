from typing import List, Optional, Union, Any
import yaml
import os
import subprocess
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import cv2
import numpy as np
from io import BytesIO
import logging
from ray.serve import get_replica_context
from PIL import Image
import time
import uuid

logger = logging.getLogger("ray.serve")

from app.api.schemas import (
    SwapUrlRequest, SwapImgRequest, ProcessingOptions, SuccessResponse, 
    PartialSuccessResponse, ErrorResponse, FailedModel
)
from app.pipe.components.analyzer import FaceAnalyzer
from app.pipe.components.enhancer import FaceEnhancer
from app.pipe.components.swapper import FaceSwapper
from app.pipe.components.enhancer_codeformer import CodeFormerEnhancer
from app.pipe.components.background_remover import BackgroundRemover
from app.pipe.swap_processor import SwapProcessor, NoSourceFaceError, NoTargetFaceError
from app.services.gcp_bucket_manager import GCPImageManager
from app.utils.utils import conditional_download

# Background removal constants
SUPPORTED_FORMATS = {"jpeg", "jpg", "png", "webp", "bmp", "tiff", "tif"}
MAX_BATCH_SIZE = 50
MAX_FILE_SIZE = 100 * 1024 * 1024

@asynccontextmanager
async def lifespan(app: FastAPI): 
    """
    Lifecycle manager for the FastAPI application.
    Downloads required models during startup.
    """
    # Load the face swap pipeline models
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
    
    # Download and convert background removal model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    onnx_model_path = os.path.join(models_dir, 'BiRefNet_dynamic-general-epoch_174_batch.onnx')
    engine_path = os.path.join(models_dir, 'engine_fp16.trt')
    
    # Step 1: Download ONNX model if not exists
    if not os.path.exists(onnx_model_path):
        logger.info("Downloading background removal ONNX model...")
        try:
            subprocess.run([
                'gdown', 
                '170qUq80CnimcWGK-VJTLEW-dVp5PI5Wd',
                '-O', onnx_model_path
            ], check=True)
            logger.info(f"ONNX model downloaded to: {onnx_model_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download ONNX model: {e}")
            raise
    else:
        logger.info(f"ONNX model already exists at: {onnx_model_path}")
    
    # Step 2: Convert ONNX to TensorRT engine if not exists
    if not os.path.exists(engine_path):
        logger.info("Converting ONNX model to TensorRT FP16 engine...")
        logger.info("This may take 1-2 minutes on first run...")
        try:
            # Import the conversion function
            from app.utils.tensorrt_utils import convert_onnx_to_engine_fp16
            
            # Convert with same settings as fp16.py
            convert_onnx_to_engine_fp16(
                onnx_filename=onnx_model_path,
                engine_filename=engine_path,
                max_batch_size=20
            )
            
            # Check engine file was created and log size
            if os.path.exists(engine_path):
                size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                logger.info(f"TensorRT engine created successfully: {engine_path}")
                logger.info(f"Engine size: {size_mb:.1f} MB")
            else:
                raise RuntimeError("Engine file was not created")
                
        except Exception as e:
            logger.error(f"Failed to convert ONNX to TensorRT engine: {e}")
            logger.error("Background removal API will not be available")
            # Don't raise - allow service to start without background removal
    else:
        logger.info(f"TensorRT engine already exists at: {engine_path}")
    
    yield
    
# Create FastAPI app with metadata
app = FastAPI(
    title="Face Swap & Background Removal API",
    description="API for face swapping with multiple modes and background removal using TensorRT",
    version="2.0.0",
    lifespan=lifespan
)

@serve.deployment()
@serve.ingress(app)
class APIIngress:
    """
    API Ingress for face swap and background removal operations using Ray Serve.
    """
    def __init__(self, 
                swap_processor: DeploymentHandle,
                img_manager: DeploymentHandle,
                background_remover: DeploymentHandle):
        """
        Initialize the API Ingress with handles to required components.
        
        Args:
            swap_processor: Handle to the SwapProcessor deployment
            img_manager: Handle to the GCPImageManager deployment
            background_remover: Handle to the BackgroundRemover deployment
        """
        self.swap_processor = swap_processor
        self.img_manager = img_manager
        self.background_remover = background_remover

    # ==================== FACE SWAP ENDPOINTS ====================

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
        """
        replica_context = get_replica_context()
        replica_tag = None if replica_context is None else replica_context.replica_tag
        
        logger.info(f"[{replica_tag}] swap_url endpoint called")
        try:
            # Extract request data
            face_filename = request.face_filename
            model_filenames = request.model_filenames
            options = request.options
                
            logger.info(f"[{replica_tag}] Request data: face={face_filename}, models={model_filenames}, options={options}")
                
            # Download source image
            logger.info(f"[{replica_tag}] Downloading source image: {face_filename}")
            source_img = await self.img_manager.download_image.remote(face_filename)
            
            img_shape = None if source_img is None else source_img.shape
            logger.info(f"[{replica_tag}] Source image download result: {img_shape}")
            
            if source_img is None:
                logger.error(f"[{replica_tag}] Failed to download image: {face_filename}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": f"Failed to download image: {face_filename}"}
                )
            
            urls = []
            failed_models = []
            
            # Process each target image
            for idx, model_filename in enumerate(model_filenames):
                logger.info(f"[{replica_tag}] Processing target image {idx+1}/{len(model_filenames)}: {model_filename}")
                try:
                    # Download target image
                    logger.info(f"[{replica_tag}] Downloading target image: {model_filename}")
                    target_img = await self.img_manager.download_image.remote(model_filename)
                    
                    img_shape = None if target_img is None else target_img.shape
                    logger.info(f"[{replica_tag}] Target image download result: {img_shape}")
                    
                    if target_img is None:
                        logger.error(f"[{replica_tag}] Failed to download target: {model_filename}")
                        failed_models.append(FailedModel(
                            filename=model_filename,
                            error="Failed to download image"
                        ))
                        continue
                    
                    # Process the swap using the SwapProcessor
                    try:
                        logger.info(f"[{replica_tag}] Calling SwapProcessor.process_swap with mode={options.mode}")
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
                        logger.info(f"[{replica_tag}] SwapProcessor.process_swap completed successfully")
                        
                    except NoSourceFaceError as e:
                        logger.error(f"[{replica_tag}] NoSourceFaceError: {str(e)}")
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"error": "Bad Request", "message": str(e)}
                        )
                    except NoTargetFaceError as e:
                        logger.error(f"[{replica_tag}] NoTargetFaceError: {str(e)}")
                        failed_models.append(FailedModel(
                            filename=model_filename,
                            error=str(e)
                        ))
                        continue
                    except Exception as e:
                        logger.exception(f"[{replica_tag}] Error in SwapProcessor: {str(e)}")
                        raise
                    
                    # Upload the result
                    logger.info(f"[{replica_tag}] Uploading result for {model_filename}")
                    url = await self.img_manager.upload_image.remote(result_img)
                    logger.info(f"[{replica_tag}] Upload result: {url}")
                    
                    if url:
                        urls.append(url)
                    else:
                        logger.error(f"[{replica_tag}] Failed to upload result for {model_filename}")
                        failed_models.append(FailedModel(
                            filename=model_filename,
                            error="Failed to upload result image"
                        ))
                    
                except HTTPException:
                    logger.exception(f"[{replica_tag}] HTTPException in processing loop")
                    raise
                except Exception as e:
                    logger.exception(f"[{replica_tag}] Unexpected error: {str(e)}")
                    failed_models.append(FailedModel(
                        filename=model_filename,
                        error=str(e)
                    ))
            
            # Determine response status code and content
            logger.info(f"[{replica_tag}] Processing complete. Successes: {len(urls)}, Failures: {len(failed_models)}")
            
            if len(failed_models) > 0 and len(urls) > 0:
                # Partial success
                logger.info(f"[{replica_tag}] Returning 206 Partial Content")
                return JSONResponse(
                    status_code=status.HTTP_206_PARTIAL_CONTENT,
                    content=jsonable_encoder(PartialSuccessResponse(urls=urls, failed=failed_models))
                )
            elif len(failed_models) > 0 and len(urls) == 0:
                # All failed
                logger.error(f"[{replica_tag}] All operations failed")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Processing Failed", "details": [jsonable_encoder(model) for model in failed_models]}
                )
            else:
                # All succeeded
                logger.info(f"[{replica_tag}] All operations succeeded")
                return SuccessResponse(urls=urls)
                
        except Exception as e:
            logger.exception(f"[{replica_tag}] Unhandled exception in swap_url: {str(e)}")
            raise

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
            source_img = self._load_image(await face.read())
            target_img = self._load_image(await model.read())
            
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
            
            img_bytes = self._result_image_bytes(result_img)
            
            return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Processing Failed", "message": str(e)}
            )

    # ==================== BACKGROUND REMOVAL ENDPOINTS ====================

    @app.post("/api/remove-bg",
              tags=["Background Removal"],
              summary="Remove background from a single image")
    async def remove_background(self,
                               image: UploadFile = File(...),
                               background: UploadFile = File(None)):
        """
        Remove background from a single uploaded image with optional background replacement.
        """
        start_time = time.perf_counter()
        
        try:
            self._validate_fast(image)
            if background and background.filename:
                self._validate_fast(background)
            
            image_data = await image.read()
            if len(image_data) == 0:
                raise HTTPException(status_code=400, detail="Empty image")
            
            input_image = Image.open(BytesIO(image_data))
            original_size = input_image.size
            
            if input_image.mode in ("CMYK", "P"):
                input_image = input_image.convert("RGB")
            
            background_image = None
            if background and background.filename:
                try:
                    bg_data = await background.read()
                    if bg_data:
                        background_image = Image.open(BytesIO(bg_data))
                        if background_image.mode == "CMYK":
                            background_image = background_image.convert("RGB")
                except Exception as e:
                    logger.error(f"Background error: {e}")
            
            # Call background remover via Ray Serve
            result_image = await self.background_remover.remove_background.remote(
                input_image, 
                background_image
            )
            
            result_size = result_image.size
            if result_size != original_size:
                logger.warning(f"Dimension mismatch - Original: {original_size}, Result: {result_size}")
            
            # Convert to bytes
            img_bytes = self._pil_to_bytes(result_image)
            
            total_time = time.perf_counter() - start_time
            
            return StreamingResponse(
                BytesIO(img_bytes),
                media_type="image/png",
                headers={
                    "X-Processing-Time": str(round(total_time, 3))
                }
            )
            
        except Exception as e:
            logger.error(f"Remove BG Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/batch-remove-bg",
              tags=["Background Removal"],
              summary="Remove backgrounds from multiple images")
    async def batch_remove_background(self,
                                     images: List[UploadFile] = File(...),
                                     backgrounds: Optional[List[UploadFile]] = File(None)):
        """
        Remove backgrounds from a batch of uploaded images with optional backgrounds.
        """
        start_time = time.perf_counter()
        
        try:
            if len(images) > MAX_BATCH_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Maximum {MAX_BATCH_SIZE} images allowed"
                )
            
            for img in images:
                self._validate_fast(img)
            
            self._validate_backgrounds(backgrounds, images)
            
            input_images = await self._load_upload_images_async(images)
            background_images = None
            
            if backgrounds:
                background_images = await self._load_upload_images_async(backgrounds)
                if len(background_images) == 1:
                    background_images = background_images * len(input_images)
            
            # Call background remover via Ray Serve
            result_images = await self.background_remover.remove_background_batch.remote(
                input_images,
                background_images
            )
            
            # Convert results to response format
            results = []
            for i, (orig_img, result_img) in enumerate(zip(input_images, result_images)):
                img_bytes = self._pil_to_bytes(result_img)
                results.append({
                    "index": i,
                    "original_filename": images[i].filename,
                    "original_size": f"{orig_img.size[0]}x{orig_img.size[1]}",
                    "result_size": f"{result_img.size[0]}x{result_img.size[1]}",
                    "data": img_bytes.hex()[:100] + "..."  # Preview only
                })
            
            total_time = time.perf_counter() - start_time
            
            return JSONResponse({
                "success": True,
                "processed_count": len(results),
                "total_count": len(images),
                "processing_time": round(total_time, 3),
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Batch Remove BG Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== HELPER METHODS ====================

    def _load_image(self, img_bytes: bytes) -> np.ndarray:
        """Load an image from bytes."""
        try:
            logger.info("_load_image called")
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.info(f"_load_image result: {'success' if img is not None else 'failed'}")
            return img
        except Exception as e:
            logger.exception(f"Error in _load_image: {str(e)}")
            return None

    def _result_image_bytes(self, image: np.ndarray) -> bytes:
        """Convert an image to bytes."""
        try:
            logger.info("_result_image_bytes called")
            _, img_encoded = cv2.imencode('.png', image)
            logger.info("_result_image_bytes encoding successful")
            return img_encoded.tobytes()
        except Exception as e:
            logger.exception(f"Error in _result_image_bytes: {str(e)}")
            raise

    def _validate_fast(self, file: UploadFile) -> None:
        """Validate uploaded file format."""
        if not file.filename or '.' not in file.filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        ext = file.filename.lower().split('.')[-1]
        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported format")

    def _validate_backgrounds(self, backgrounds, images):
        """Validate background images for batch processing."""
        if backgrounds:
            if len(backgrounds) != 1 and len(backgrounds) != len(images):
                raise HTTPException(
                    status_code=400, 
                    detail="Backgrounds must be 1 or equal to number of images"
                )
            for bg in backgrounds:
                self._validate_fast(bg)

    async def _load_upload_images_async(self, files: List[UploadFile]) -> List[Image.Image]:
        """Asynchronously load uploaded images."""
        images = []
        for file in files:
            img_data = await file.read()
            if len(img_data) == 0:
                continue
            img = Image.open(BytesIO(img_data))
            if img.mode in ("CMYK", "P"):
                img = img.convert("RGB")
            images.append(img)
        return images

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        buf = BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue()


# Load the GCP Bucket config
with open('app/configs/serve_config_gpu.yaml', 'r') as file:
    config = yaml.safe_load(file)

gcp_image_manager_config = None
for app_config in config['applications']:
    if app_config['name'] == 'SwapFaceAPI':
        for deployment in app_config['deployments']:
            if deployment['name'] == 'GCPImageManager':
                gcp_image_manager_config = deployment['init_args']

if gcp_image_manager_config is None:
    raise ValueError("GCPImageManager configuration not found in config.yaml")

# Bind actors with configuration
analyzer = FaceAnalyzer.bind()
swapper = FaceSwapper.bind()
enhancer = FaceEnhancer.bind()
codeformer = CodeFormerEnhancer.bind()
img_manager = GCPImageManager.bind(*gcp_image_manager_config)

# Create the SwapProcessor
swap_processor = SwapProcessor.bind(analyzer, swapper, enhancer, codeformer)

# Create the BackgroundRemover
background_remover = BackgroundRemover.bind()

# Bind API Ingress with all components
app = APIIngress.bind(swap_processor, img_manager, background_remover)