import io
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from .utils import get_nparray_from_uploadfile, add_watermark, apply_vignette
from model import swap_face

router = APIRouter()

@router.post("/swap_face")
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
