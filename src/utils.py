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


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


async def async_conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
        
    async with httpx.AsyncClient() as client:
        for url in urls:
            download_file_path = os.path.join(download_directory_path, os.path.basename(url))
            if not os.path.exists(download_file_path):
                # Perform the async request
                response = await client.get(url)
                response.raise_for_status()  # Ensure we got a valid response
                
                # Get content length for progress bar, defaulting to 0 if not found
                total = int(response.headers.get('Content-Length', 0))
                
                # Write response content to file with tqdm progress bar
                # Open the file outside of the progress context to avoid closing it on each update
                with open(download_file_path, 'wb') as f, tqdm(
                    total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024
                ) as progress:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        progress.update(len(chunk))


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


async def read_image_as_array(image_file: UploadFile) -> np.ndarray:
    image_bytes = await image_file.read()
    image = Image.open(BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)