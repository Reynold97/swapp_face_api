import mimetypes
import urllib
from typing import List, Optional
from tqdm import tqdm
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi import UploadFile
import httpx
import asyncio
from dotenv import load_dotenv

load_dotenv(".env")

def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    token = os.getenv("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)

    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            req = urllib.request.Request(url, headers=headers)  # Add headers to the request
            with urllib.request.urlopen(req) as response, open(download_file_path, 'wb') as out_file:
                data = response.read()  # Read the response data
                out_file.write(data)
            total = int(response.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                progress.update(total)


async def async_conditional_download(download_directory_path: str, urls: List[str]) -> None:
    token = os.getenv("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
        
    async with httpx.AsyncClient() as client:
        for url in urls:
            download_file_path = os.path.join(download_directory_path, os.path.basename(url))
            if not os.path.exists(download_file_path):
                response = await client.get(url, headers=headers)  # Include headers in async request
                response.raise_for_status()
                
                total = int(response.headers.get('content-length', 0))
                with open(download_file_path, 'wb') as f, tqdm(
                    total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                    f.write(response.content)
                    progress.update(total)
                    

def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


async def read_image_as_array(image_file: UploadFile) -> np.ndarray:
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)