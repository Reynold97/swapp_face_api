import numpy as np
from typing import List

from src.pipe.components.analyzer import FaceAnalyzer
from src.pipe.components.swapper import FaceSwapper
from src.pipe.components.enhancer import FaceEnhancer
from src.typing import Frame

class ImagePipeline:
    def __init__(self, providers: List[str]):
        self.face_analyzer = FaceAnalyzer(providers)
        self.face_swapper = FaceSwapper(providers)
        self.face_enhancer = FaceEnhancer(providers)

    def process_1_image_default_face(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:        
        #swapp process
        source_face = self.face_analyzer.get_one_face(source)
        reference_face = self.face_analyzer.get_one_face(target) 
        swapped_frame = self.face_swapper.swap_face(source_face, reference_face, target) 
        
        #enhance process      
        many_faces = self.face_analyzer.get_many_faces(swapped_frame)
        for target_face in many_faces:
            enhanced_frame = self.face_enhancer.enhance_face(target_face, swapped_frame)
        return enhanced_frame
       
