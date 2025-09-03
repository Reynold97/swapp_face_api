from typing import List, Optional, Union, Any
import yaml
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import cv2
import numpy as np
from io import BytesIO
import logging
logger = logging.getLogger("ray.serve")

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
    
# Initialize processors directly without Ray Serve
analyzer = FaceAnalyzer()
swapper = FaceSwapper()
enhancer = FaceEnhancer()
codeformer = CodeFormerEnhancer()
img_manager = GCPImageManager("/app/credentials/bucket_credentials.json", "anyscale_tmp_faces")

# Create the SwapProcessor
swap_processor = SwapProcessor(analyzer, swapper, enhancer, codeformer)

# Create FastAPI app with metadata
app = FastAPI(
    title="Face Swap API",
    description="API for face swapping with multiple modes and enhancement options",
    version="2.0.0",
    lifespan=lifespan
)

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
async def swap_url(request: SwapUrlRequest):
        """
        Perform face swapping using images stored in GCP bucket.
        """

        try:
            # Extract request data
            face_filename = request.face_filename
            model_filenames = request.model_filenames
            options = request.options
                
            source_img = img_manager.download_image(face_filename)
            
            img_shape = None if source_img is None else source_img.shape

            if source_img is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Bad Request", "message": f"Failed to download image: {face_filename}"}
                )
            
            urls = []
            failed_models = []
            
            # Process each target image
            for idx, model_filename in enumerate(model_filenames):
                try:
                    # Download target image
                    target_img = img_manager.download_image(model_filename)
                    
                    img_shape = None if target_img is None else target_img.shape
                    
                    if target_img is None:
                        failed_models.append(FailedModel(
                            filename=model_filename,
                            error="Failed to download image"
                        ))
                        continue
                    
                    # Process the swap using the SwapProcessor
                    try:
                        result_img = await swap_processor.process_swap(
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
                        failed_models.append(FailedModel(
                            filename=model_filename,
                            error=str(e)
                        ))
                        continue
                    except Exception as e:
                        raise
                    
                    # Upload the result
                    url = img_manager.upload_image(result_img)
                    
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
                
        except Exception as e:

            raise

@app.post("/swap_img",
          responses={
              200: {"description": "Successful operation", "content": {"image/png": {}}},
              400: {"model": ErrorResponse, "description": "Bad request"},
              500: {"model": ErrorResponse, "description": "Server error"}
          },
          tags=["Face Swap"],
          summary="Swap faces using uploaded images")
async def swap_img(model: UploadFile = File(..., description="Target image file"),
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
            source_img = __load_image(await face.read())
            target_img = __load_image(await model.read())
            
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
                result_img = await swap_processor.process_swap(
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
            img_bytes = __result_image_bytes(result_img)
            
            return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Processing Failed", "message": str(e)}
            )

def __load_image(img_bytes: bytes) -> np.ndarray:
    """Load an image from bytes."""
    try:
        logger.info("__load_image called")
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        logger.info(f"__load_image result: {'success' if img is not None else 'failed'}")
        return img
    except Exception as e:
        logger.exception(f"Error in __load_image: {str(e)}")
        return None

def __result_image_bytes(image: np.ndarray) -> bytes:
    """Convert an image to bytes."""
    try:
        logger.info("__result_image_bytes called")
        _, img_encoded = cv2.imencode('.png', image)
        logger.info("__result_image_bytes encoding successful")
        return img_encoded.tobytes()
    except Exception as e:
        logger.exception(f"Error in __result_image_bytes: {str(e)}")
        raise