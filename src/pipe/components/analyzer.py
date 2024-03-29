import threading
from typing import Any, Optional, List
import numpy

import insightface
from insightface.app.common import Face

from src.typing import Frame
from src.pipe.components.processor import Processor


class FaceAnalyzer(Processor):      
    def __init__(self, providers: List[str]):
        super().__init__('FACE_ANALYSER', '', providers)  # model path is blank because is downloaded from insightface, not local.
        self.load_model() # Ensures model is loaded during instantiation
       
    def load_model(self) -> Any:
        with threading.Lock():
            if self.model is None:
                self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=self.providers)
                self.model.prepare(ctx_id=0)
        return self.model

    def get_one_face(self, frame: Frame, position: int = 0) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            try:
                return many_faces[position]
            except IndexError:
                return many_faces[-1]
        return None

    def get_many_faces(self, frame: Frame) -> Optional[List[Face]]:
        try:
            return self.model.get(frame)
        except ValueError:
            return None

    def find_similar_face(self, frame: Frame, reference_face: Face, similar_face_distance: float) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            for face in many_faces:
                if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                    distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                    if distance < similar_face_distance:
                        return face
        return None