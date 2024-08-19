import os
import requests
import numpy
from typing import Any
from collections import namedtuple


# Define a named tuple 'Face' with fields 'kps' and 'embedding'
Face = namedtuple('Face', ['kps', 'embedding'])

# Define Frame, background image
Frame = numpy.ndarray[Any, Any]

def conditional_download(url: str, root_folder: str = './.assets') -> str:
    """
    Downloads a file from the given URL if it does not already exist in the specified folder.
    
    Args:
        url (str): The URL of the file to download.
        root_folder (str, optional): The folder to save the downloaded file. Defaults to './.assets'.
    
    Returns:
        str: The path to the downloaded file.
    
    Raises:
        Exception: If the file cannot be downloaded.
    """
    model_filename = url.split('/')[-1]
    model_path = os.path.join(root_folder, model_filename)

    # Create the root folder if it doesn't exist
    os.makedirs(root_folder, exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.exists(model_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Cannot download the model {model_filename}")

    return model_path
