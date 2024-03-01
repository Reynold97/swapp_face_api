import io
import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from model import swap_face, swap_video
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from .utils import get_nparray_from_uploadfile, add_watermark, apply_vignette

router = APIRouter()

@router.post("/swap_face/img")
async def swap(model: UploadFile = File(...), face: UploadFile = File(...), watermark: bool = Form(...), vignette: bool = Form(...)):
    try:
        # Convert the uploaded images to NumPy arrays
        model = await get_nparray_from_uploadfile(model)
        face = await get_nparray_from_uploadfile(face)

        result = swap_face(model, face)
        
        if vignette:
          result = apply_vignette(result)
        if watermark:
          result = add_watermark(result)

        # Convert from BGR to RGB
        result = result[:, :, ::-1]  

        # Convertir el array en una imagen
        image = Image.fromarray(result)
        
        # Convert PIL Image to a byte stream (in memory)
        byte_io = io.BytesIO()
        image.save(byte_io, format="PNG")
        byte_io.seek(0)  # Go back to the start of the bytes stream

        # Return image as a stream
        return StreamingResponse(byte_io, media_type="image/png")

    except Exception as e:
        # In case of an error, return an error response
        return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=500)


@router.post("/swap_face/video")
async def swap__video(video: UploadFile = File(...), face: UploadFile = File(...)):
    # Ensure the uploads directory exists
    folder_name = '/roop-api/.tmp'
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    Path(folder_name).mkdir(exist_ok=True)

    # Construct the path to save the uploaded file
    video_path = f'{folder_name}/{video.filename}'
    img_path = f'{folder_name}/{face.filename}'
    output_path = f'{folder_name}/swapped.mp4'

    # Save the uploaded file to disk
    with open(video_path, "wb+") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(img_path, "wb+") as buffer:
        shutil.copyfileobj(face.file, buffer)

    swap_video(video_path, img_path, output_path)

    return FileResponse(output_path, media_type="video/mp4", status_code=200)

