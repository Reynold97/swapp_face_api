from typing import List
import yaml
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import cv2
import numpy as np
from io import BytesIO
from app.pipe.components.analyzer import FaceAnalyzer
from app.pipe.components.enhancer import FaceEnhancer
from app.pipe.components.swapper import FaceSwapper
from app.pipe.components.enhancer_codeformer import CodeFormerEnhancer
from app.services.gcp_bucket_manager import GCPImageManager
from app.utils.utils import conditional_download

@asynccontextmanager
async def lifespan(app: FastAPI): 
    # Load the pipeline models
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/inswapper_128_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/arcface_w600k_r50_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/retinaface_10g_fp16.onnx')
    conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/gfpgan_1.4.onnx')
    #CodeFormer Models
    # Download detection and parsing models if needed
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
    
app = FastAPI(lifespan=lifespan)

@serve.deployment()
@serve.ingress(app)
class APIIngress:
    """
    A API class that sevres as ingress to the ray service and handles the image processing pipeline including face analysis, 
    swapping, and enhancement.
    """
    def __init__(self, analyzer_handle: DeploymentHandle, swapper_handle: DeploymentHandle, enhancer_handle: DeploymentHandle, codeformer_handle: DeploymentHandle, img_manager: DeploymentHandle):
        """
        Initializes the ImagePipeline class with deployment handles for analyzer, swapper, enhancer, and image downloader.
        
        Args:
            analyzer_handle (`DeploymentHandle`): The deployment handle for the FaceAnalyzer.
            swapper_handle (`DeploymentHandle`): The deployment handle for the FaceSwapper.
            enhancer_handle (`DeploymentHandle`): The deployment handle for the FaceEnhancer.
            codeformer_handle (`DeploymentHandle`): The deployment handle for the CodeFormerEnhancer.
            img_downloader (`DeploymentHandle`): The deployment handle for the image downloader.
        """
        self.analyzer_handle = analyzer_handle
        self.swapper_handle = swapper_handle
        self.enhancer_handle = enhancer_handle
        self.codeformer_handle = codeformer_handle
        self.img_manager = img_manager

    @app.post("/swap_url")
    async def swap_url(self, request: Request) -> JSONResponse:
        """
        Handles face swapping for images specified by URLs.
        
        Args:
            request (`Request`): The incoming request containing model filenames and face filename.
        
        Returns:
            `dict`: A dictionary containing the URLs of the processed images.
        """
        request_data = await request.json()
        model_filenames, face_filename = request_data["model_filenames"], request_data["face_filename"]

        source = await self.img_manager.download_image.remote(face_filename)
        source_face = await self.analyzer_handle.extract_faces.remote(source)

        if source_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `face_filename`."}, status_code=400)

        urls = []

        for model_filename in model_filenames:
            target = await self.img_manager.download_image.remote(model_filename)
            target_face = await self.analyzer_handle.extract_faces.remote(target)

            tmp = await self.swapper_handle.swap_face.remote(source_face, target_face, target)
            target_face = await self.analyzer_handle.extract_faces.remote(tmp)
            tmp = await self.enhancer_handle.enhance_face.remote(target_face, tmp)

            url = await self.img_manager.upload_image.remote(tmp)
            urls.append(url)

        partial_success = False
        for i, url in enumerate(urls):
            if urls[i] is None:
                partial_success = True

        return JSONResponse({"urls": urls}, status_code=200 if not partial_success else 206)
    
    @app.post("/swap_url2")
    async def swap_url2(self, face_filename: str, model_filenames: List[str]) -> JSONResponse:
        """
        Handles face swapping for images specified by URLs.
        
        Args:
            face_filename: str  The Face file name in the bucket
            model_filenames: List[str]  List of the models file names in the bucket
        
        Returns:
            `dict`: A dictionary containing the URLs of the processed images.
        """
        face_filename = face_filename
        model_filenames= model_filenames
        
        source = await self.img_manager.download_image.remote(face_filename)
        source_face = await self.analyzer_handle.extract_faces.remote(source)

        if source_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `face_filename`."}, status_code=400)

        urls = []

        for model_filename in model_filenames:
            target = await self.img_manager.download_image.remote(model_filename)
            target_face = await self.analyzer_handle.extract_faces.remote(target)

            tmp = await self.swapper_handle.swap_face.remote(source_face, target_face, target)
            target_face = await self.analyzer_handle.extract_faces.remote(tmp)
            tmp = await self.enhancer_handle.enhance_face.remote(target_face, tmp)

            url = await self.img_manager.upload_image.remote(tmp)
            urls.append(url)

        partial_success = False
        for i, url in enumerate(urls):
            if urls[i] is None:
                partial_success = True

        return JSONResponse({"urls": urls}, status_code=200 if not partial_success else 206)

    @app.post("/swap_img")
    async def swap_img(self, model: UploadFile, face: UploadFile): #-> StreamingResponse:
        """
        Handles face swapping for uploaded images.
        
        Args:
            model: UploadFile  Model image file
            face: UploadFile   Face image file
        
        Returns:
            `StreamingResponse`: The response containing the processed image.
        """
        source = self.__load_image(await face.read())
        target = self.__load_image(await model.read())

        source_face = await self.analyzer_handle.extract_faces.remote(source)
        if source_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `face`."}, status_code=400)

        target_face = await self.analyzer_handle.extract_faces.remote(target)
        if target_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `model`."}, status_code=400)

        tmp = await self.swapper_handle.swap_face.remote(source_face, target_face, target)
        target_face = await self.analyzer_handle.extract_faces.remote(tmp)
        tmp = await self.enhancer_handle.enhance_face.remote(target_face, tmp)

        result_img = tmp
        img_bytes = self.__result_image_bytes(result_img)

        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
    
    @app.post("/swap_url_codeformer")
    async def swap_url_codeformer(self, face_filename: str, model_filenames: List[str], fidelity_weight: float = 0.5, background_enhance: bool = True, face_upsample: bool = True) -> JSONResponse:
        """
        Handles face swapping with CodeFormer enhancement for images specified by URLs.
        
        Args:
            face_filename: str  The Face file name in the bucket
            model_filenames: List[str]  List of the models file names in the bucket
            fidelity_weight: float  Balance quality and fidelity (0: better quality, 1: better identity)
            background_enhance: bool  Whether to enhance the background with RealESRGAN
            face_upsample: bool  Whether to upsample the face with RealESRGAN
        
        Returns:
            `dict`: A dictionary containing the URLs of the processed images.
        """
        source = await self.img_manager.download_image.remote(face_filename)
        source_face = await self.analyzer_handle.extract_faces.remote(source)

        if source_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `face_filename`."}, status_code=400)

        urls = []

        for model_filename in model_filenames:
            target = await self.img_manager.download_image.remote(model_filename)
            target_face = await self.analyzer_handle.extract_faces.remote(target)

            # Swap face
            tmp = await self.swapper_handle.swap_face.remote(source_face, target_face, target)
            
            # Extract face from swapped image
            target_face = await self.analyzer_handle.extract_faces.remote(tmp)
            
            # Enhance with CodeFormer instead of regular enhancer
            tmp = await self.codeformer_handle.enhance_face.remote(
                target_face, 
                tmp, 
                fidelity_weight, 
                background_enhance, 
                face_upsample, 
                2  # Fixed upscale factor of 2
            )

            url = await self.img_manager.upload_image.remote(tmp)
            urls.append(url)

        partial_success = False
        for i, url in enumerate(urls):
            if urls[i] is None:
                partial_success = True

        return JSONResponse({"urls": urls}, status_code=200 if not partial_success else 206)

    def __load_image(self, img_bytes: bytes) -> np.ndarray:
        """
        Loads an image from bytes.
        
        Args:
            img_bytes (`bytes`): The image bytes.
        
        Returns:
            `np.ndarray`: The loaded image.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def __result_image_bytes(self, image: np.ndarray) -> bytes:
        """
        Encodes an image to bytes.
        
        Args:
            image (`np.ndarray`): The image to encode.
        
        Returns:
            `bytes`: The encoded image bytes.
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

# Bind actors with configuration
analyzer = FaceAnalyzer.bind()
swapper = FaceSwapper.bind()
enhancer = FaceEnhancer.bind()
codeformer = CodeFormerEnhancer.bind()
img_manager = GCPImageManager.bind(*gcp_image_manager_config)

# Bind API Gateway with actors
app = APIIngress.bind(analyzer, swapper, enhancer, codeformer, img_manager)