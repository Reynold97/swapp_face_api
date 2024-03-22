from typing import List
import threading

from gfpgan.utils import GFPGANer
from insightface.app.common import Face

import src.globals
from src.pipe.components.processor import Processor
from src.typing import Frame


class FaceEnhancer(Processor):

    def __init__(self, providers: List[str]):
        super().__init__('FACE-ENHANCER','../models/GFPGANv1.4.pth', providers)
        self.load_model()  # Ensures model is loaded during instantiation 
        
    def load_model(self):
        if not self.model:
            # Thread-safe operation 
            with threading.Lock():
                self.model = GFPGANer(model_path=self.model_path, upscale=1, device=self.get_device())
    
    def get_device(self) -> str:
        if 'CUDAExecutionProvider' in self.providers:
            return 'cuda'
        if 'CoreMLExecutionProvider' in self.providers:
            return 'mps'
        return 'cpu'
        
    def enhance_face(self, target_face: Face, temp_frame: Frame) -> Frame:
        start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
        padding_x = int((end_x - start_x) * 0.5)
        padding_y = int((end_y - start_y) * 0.5)
        start_x = max(0, start_x - padding_x)
        start_y = max(0, start_y - padding_y)
        end_x = max(0, end_x + padding_x)
        end_y = max(0, end_y + padding_y)
        temp_face = temp_frame[start_y:end_y, start_x:end_x]
        if temp_face.size:
            with threading.Semaphore():
                _, _, temp_face = self.model.enhance(
                    temp_face,
                    paste_back=True
                )
            temp_frame[start_y:end_y, start_x:end_x] = temp_face
        return temp_frame

