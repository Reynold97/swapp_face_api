from typing import Any, Optional
import cv2
import numpy as np
import threading

import insightface
from insightface.app.common import Face

from src.utils import resolve_relative_path, conditional_download
#from src.pipe.components.analyzer import get_one_face, get_many_faces, find_similar_face
from src.pipe.components.processor import Processor
import src.globals
from src.typing import Frame
from src.pipe.components.analyzer import FaceAnalyzer


class FaceSwapper(Processor):
    def __init__(self):        
        super().__init__('FACE-SWAPPER', '../models/inswapper_128.onnx')
        self.load_model()  # Ensures model is loaded during instantiation
        self.face_analyzer = FaceAnalyzer()
    
    def load_model(self):
        if not self.model:
            # Thread-safe operation 
            with threading.Lock():
                self.model = insightface.model_zoo.get_model(self.model_path, providers=src.globals.execution_providers)

    def clear_model(self) -> None:
        self.model=None

    def post_process(self) -> None:
        self.clear_model()

    def swap_face(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        return self.model.get(temp_frame, target_face, source_face, paste_back=True)
    
    def process_frame(self, source_face: Optional[Face], reference_face: Optional[Face], temp_frame: Frame) -> Frame:
        if src.globals.many_faces:
            many_faces = self.face_analyzer.get_many_faces(temp_frame)
            for target_face in many_faces:
                temp_frame = self.swap_face(source_face, target_face, temp_frame)
        else:
            target_face = self.face_analyzer.find_similar_face(temp_frame, reference_face)
            temp_frame = self.swap_face(source_face, target_face, temp_frame)
        return temp_frame
    
    def process_image(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_face = self.face_analyzer.get_one_face(source)
        target_frame = target
        reference_face = None if src.globals.many_faces else self.face_analyzer.get_one_face(target_frame, src.globals.reference_face_position)
        result = self.process_frame(source_face, reference_face, target_frame)
        #cv2.imwrite(output_path, result)
        return result