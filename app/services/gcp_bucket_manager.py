import cv2
import uuid
import numpy as np
from ray import serve
from google.cloud import storage
from google.oauth2.service_account import Credentials


@serve.deployment()
class GCPBucketManager:
    """
    Manages the upload and download of images to/from a GCP bucket.
    
    This class uses OpenCV for image manipulation, converting numpy arrays
    to images for upload, and images to numpy arrays for download.
    """

    def __init__(self, credentials_path, bucket_name):
        """
        Initializes the image manager with GCP credentials and bucket.
        
        :param credentials_path: Path to the GCP credentials JSON file.
        :param bucket_name: Name of the GCP bucket for image storage.
        """
        self.credentials = Credentials.from_service_account_file(credentials_path)
        self.client = storage.Client(credentials=self.credentials)
        self.bucket = self.client.bucket(bucket_name)

    def upload_image(self, image_array):
        """
        Uploads an image, represented as a numpy array, to the GCP bucket.
        
        The image is converted to JPEG format before uploading. A unique UUID
        is used as the image name.
        
        :param image_array: Numpy array representing the image.
        :return: The unique filename of the uploaded image.
        """
        if image_array is None:
            return None
        _, buffer = cv2.imencode('.jpg', image_array)
        file_name = f"{uuid.uuid4()}.jpg"
        blob = self.bucket.blob(file_name)
        blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
        return file_name

    def download_image(self, file_name):
        """
        Downloads an image from the GCP bucket and converts it to a numpy array.
        
        :param file_name: The unique filename of the image to download.
        :return: Numpy array representing the downloaded image.
        """
        blob = self.bucket.blob(file_name)
        image_bytes = blob.download_as_bytes()
        image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image_array
