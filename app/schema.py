from pydantic import BaseModel


class ImageProcessRequest(BaseModel):
    source_path: str
    target_path: str
    output_path: str