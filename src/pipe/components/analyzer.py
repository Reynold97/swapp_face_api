import threading
from typing import Any, Optional, List
import numpy

import insightface
from insightface.app.common import Face

from src.typing import Frame
import src.globals


class FaceAnalyzer:
    def __init__(self):
        self.name = 'FACE_ANALYSER'
        self.model = self.load_model() # Ensures model is loaded during instantiation
        #self.lock = threading.Lock()  # Ensure thread-safe operations

    def load_model(self) -> Any:
        with threading.Lock():
            if self.model is None:
                self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=src.globals.execution_providers)
                self.model.prepare(ctx_id=0)
        return self.model

    def clear_face_analyser(self) -> None:
        self.model = None

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
            return self.get_face_analyser().get(frame)
        except ValueError:
            return None

    def find_similar_face(self, frame: Frame, reference_face: Face) -> Optional[Face]:
        many_faces = self.get_many_faces(frame)
        if many_faces:
            for face in many_faces:
                if hasattr(face, 'normed_embedding') and hasattr(reference_face, 'normed_embedding'):
                    distance = numpy.sum(numpy.square(face.normed_embedding - reference_face.normed_embedding))
                    if distance < src.globals.similar_face_distance:
                        return face
        return None