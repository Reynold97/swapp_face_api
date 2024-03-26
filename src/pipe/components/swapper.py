from typing import List
import threading

import insightface
from insightface.app.common import Face

from src.pipe.components.processor import Processor
from src.typing import Frame


class FaceSwapper(Processor):
    def __init__(self, providers: List[str]):        
        super().__init__('FACE-SWAPPER', '../models/inswapper_128.onnx', providers)
        self.load_model()  # Ensures model is loaded during instantiation
    
    def load_model(self):
        if not self.model:
            # Thread-safe operation 
            with threading.Lock():
                self.model = insightface.model_zoo.get_model(self.model_path, providers=self.providers)

    def swap_face(self, source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
        return self.model.get(temp_frame, target_face, source_face, paste_back=True)
    
    