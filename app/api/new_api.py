import ray
from ray import serve
from fastapi import FastAPI, Request, Form, UploadFile
from starlette.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class APIGateway:
    def __init__(self, analyzer_handle: serve.DeploymentHandle, swapper_handle: serve.DeploymentHandle, enhancer_handle: serve.DeploymentHandle, img_downloader: serve.DeploymentHandle):
        self.analyzer_handle = analyzer_handle
        self.swapper_handle = swapper_handle
        self.enhancer_handle = enhancer_handle
        self.img_downloader = img_downloader

    @app.post("/swap_url")
    async def swap_url(self, request: Request) -> JSONResponse:
        request_data = await request.json()
        model_filenames, face_filename = request_data["model_filenames"], request_data["face_filename"]

        source = await self.img_downloader.download_image.remote(face_filename)
        source_face = await self.analyzer_handle.extract_faces.remote(source)

        if source_face is None:
            return JSONResponse({"error": "Bad Request", "message": "No face detected in the provided `face_filename`."}, status_code=400)

        urls = []

        for model_filename in model_filenames:
            target = await self.img_downloader.download_image.remote(model_filename)
            target_face = await self.analyzer_handle.extract_faces.remote(target)

            tmp = await self.swapper_handle.swap_face.remote(source_face, target_face, target)
            target_face = await self.analyzer_handle.extract_faces.remote(tmp)
            tmp = await self.enhancer_handle.enhance_face.remote(target_face, tmp)

            url = await self.img_downloader.upload_image.remote(tmp)
            urls.append(url)

        partial_success = False
        for i, url in enumerate(urls):
            if urls[i] is None:
                partial_success = True

        return JSONResponse({"urls": urls}, status_code=200 if not partial_success else 206)

    @app.post("/swap_img")
    async def swap_img(self, model: UploadFile, face: UploadFile) -> StreamingResponse:
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

        result_img = await tmp
        img_bytes = self.__result_image_bytes(result_img)

        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    def __load_image(self, img_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def __result_image_bytes(self, image: np.ndarray) -> bytes:
        _, img_encoded = cv2.imencode('.png', image)
        return img_encoded.tobytes()

# Run the Ray Serve deployment with FastAPI
serve.run(APIGateway.bind(analyzer_handle, swapper_handle, enhancer_handle, img_downloader))
