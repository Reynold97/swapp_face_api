from abc import ABC, abstractmethod
import threading
from typing import Any, Optional
import cv2
import numpy

from insightface.app.common import Face

from src.typing import Frame
from src.utils import resolve_relative_path


class Processor(ABC):
    
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.model_path = resolve_relative_path(model_path) # Path to the model
        self.model = None  # Placeholder for the model instance

    @abstractmethod
    def load_model(self) -> Any:
        pass

    @abstractmethod
    def clear_model(self) -> None:
        pass

    @abstractmethod
    def post_process(self) -> None:
        pass

    @abstractmethod
    def process_frame(self, source_face: Optional[Face], reference_face: Optional[Face], temp_frame: Frame) -> Frame:
        pass
    
    @abstractmethod
    def process_image(self, source_img: Any, target_path: Any) -> Any:
        pass