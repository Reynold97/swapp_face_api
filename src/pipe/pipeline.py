import numpy as np
from typing import List

from src.pipe.components.analyzer import FaceAnalyzer
from src.pipe.components.swapper import FaceSwapper
from src.pipe.components.enhancer import FaceEnhancer
import src.globals
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
       
        
    

"""
    #Legacy functions 
    #Swapper
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
    
    #Enhancer
    def process_frame(self, source_face: Optional[Face], reference_face: Optional[Face], temp_frame: Frame) -> Frame:
        many_faces = self.face_analyzer.get_many_faces(temp_frame)
        for target_face in many_faces:
            temp_frame = self.enhance_face(target_face, temp_frame)
        return temp_frame
    
    def process_image(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        target_frame = target
        result = self.process_frame(None, None, target_frame)
        #cv2.imwrite(output_path, result)
        return result
"""