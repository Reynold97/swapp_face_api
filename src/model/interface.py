import cv2
import numpy as np

from roop import globals
from roop.face_analyser import get_one_face
from roop.core import decode_execution_providers
import roop.processors.frame.face_swapper as swapper
import roop.processors.frame.face_enhancer as enhancer

swapper.pre_check()
enhancer.pre_check()

globals.execution_providers = decode_execution_providers(['cpu'])
globals.many_faces = True

def swap_face(model: np.ndarray, face: np.ndarray) -> np.ndarray:
  source_face = get_one_face(face)
  target_frame = model
  reference_face = None # if model.roop.globals.many_faces else get_one_face(target_frame, model.roop.globals.reference_face_position)
  inswapped = swapper.process_frame(source_face, reference_face, target_frame)
  return enhancer.process_frame(None, None, inswapped)

if __name__ == '__main__':
  model = cv2.imread('./.github/examples/samurai.png')
  face  = cv2.imread('./.github/examples/Reynold_Oramas.jpg')
  globals.source_path=model 
  globals.target_path=face
  globals.output_path='./.github/examples/output.png'
  inswapped = swap_face(model, face)
  cv2.imwrite('./.github/examples/output.png', inswapped)
  print('AAAAAAAAA')