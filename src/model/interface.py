import cv2
import numpy as np


import model.roop.globals
from model.roop.predictor import predict_video
from model.roop.face_analyser import get_one_face
from model.roop.core import destroy
from model.roop.core import decode_execution_providers
import model.roop.processors.frame.face_swapper as swapper
import model.roop.processors.frame.face_enhancer as enhancer
from model.roop.processors.frame.core import get_frame_processors_modules
from model.roop.utilities import is_video, detect_fps, create_video, \
  extract_frames, get_temp_frame_paths, restore_audio, create_temp,  \
  move_temp, clean_temp


swapper.pre_check()
enhancer.pre_check()

model.roop.globals.execution_providers = decode_execution_providers(['cuda'])
model.roop.globals.many_faces = True

def swap_face(model: np.ndarray, face: np.ndarray) -> np.ndarray:
  source_face = get_one_face(face)
  target_frame = model
  reference_face = None # if model.roop.globals.many_faces else get_one_face(target_frame, model.roop.globals.reference_face_position)
  inswapped = swapper.process_frame(source_face, reference_face, target_frame)
  return enhancer.process_frame(None, None, inswapped)


def swap_video(video_path, face_path, output_path):
    model.roop.globals.keep_fps = True
    model.roop.globals.skip_audio = True
    model.roop.globals.keep_frames = True
    model.roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    model.roop.globals.source_path = face_path
    model.roop.globals.target_path = video_path
    model.roop.globals.output_path = output_path
    model.roop.globals.temp_frame_format = 'jpg'
    model.roop.globals.temp_frame_quality = 0
    model.roop.globals.output_video_encoder = 'libx264'
    model.roop.globals.output_video_quality = 35
    model.roop.globals.execution_threads = 8


    # process image to videos
    if predict_video(model.roop.globals.target_path):
        destroy()
    #update_status('Creating temporary resources...')
    create_temp(model.roop.globals.target_path)
    # extract frames
    if model.roop.globals.keep_fps:
        fps = detect_fps(model.roop.globals.target_path)
        #update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(model.roop.globals.target_path, fps)
    else:
        #update_status('Extracting frames with 30 FPS...')
        extract_frames(model.roop.globals.target_path)
    # process frame
    temp_frame_paths = get_temp_frame_paths(model.roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(model.roop.globals.frame_processors):
            #update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(model.roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        #update_status('Frames not found...')
        return
    # create video
    if model.roop.globals.keep_fps:
        fps = detect_fps(model.roop.globals.target_path)
        #update_status(f'Creating video with {fps} FPS...')
        create_video(model.roop.globals.target_path, fps)
    else:
        #update_status('Creating video with 30 FPS...')
        create_video(model.roop.globals.target_path)
    # handle audio
    if model.roop.globals.skip_audio:
        move_temp(model.roop.globals.target_path, model.roop.globals.output_path)
        #update_status('Skipping audio...')
    else:
        # if model.roop.globals.keep_fps:
        #     #update_status('Restoring audio...')
        # else:
        #     #update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(model.roop.globals.target_path, model.roop.globals.output_path)
    # clean temp
    #update_status('Cleaning temporary resources...')
    clean_temp(model.roop.globals.target_path)
    # validate video
    # if is_video(model.roop.globals.target_path):
    #     #update_status('Processing to video succeed!')
    # else:
    #     #update_status('Processing to video failed!')

if __name__ == '__main__':
  model = cv2.imread('./.github/examples/samurai.png')
  face  = cv2.imread('./.github/examples/Reynold_Oramas.jpg')
  model.roop.globals.source_path=model 
  model.roop.globals.target_path=face
  model.roop.globals.output_path='./.github/examples/output.png'
  inswapped = swap_face(model, face)
  cv2.imwrite('./.github/examples/output.png', inswapped)
  print('AAAAAAAAA')