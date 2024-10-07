import cv2
import cupy as cp
import numpy as np
import onnxruntime
from ray import serve
from app.utils.utils import conditional_download


################################
#########   ENHANCER   #########
################################

@serve.deployment()
class FaceEnhancer:
    """
    A class used to enhance faces in a given frame using GFPGAN v1.4 model.
    """

    def __init__(self) -> None:
        """
        Initializes the FaceEnhancer class by downloading and setting up the ONNX model for face enhancement.
        """
        self.enhancer = onnxruntime.InferenceSession(
            conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/gfpgan_1.4.onnx'),
            providers=['CUDAExecutionProvider']
        )
        
        print(f'ENHANCER is using the following provider(s): {self.enhancer.get_providers()}')

    def enhance_face(self, target_face, temp_frame: np.ndarray) -> np.ndarray:
        """
        Enhances the given face in the frame and returns the enhanced frame.
        
        Args:
            target_face (`Face`): The target face namedtuple containing keypoints and embedding.
            temp_frame (`np.ndarray`): The input image frame.
        
        Returns:
            `np.ndarray`: The enhanced frame.
        """
        if target_face is None:
            return None

        model_template = np.array([
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465]
        ])
        model_size = (512, 512)

        crop_frame, affine_matrix = self.__warp_face_by_kps(temp_frame, target_face.kps, model_template, model_size)
        crop_mask_list = [
            self.__create_static_box_mask(crop_frame.shape[:2][::-1], 0.3, (0, 0, 0, 0))
        ]
        crop_frame = self.__prepare_crop_frame_enhance(crop_frame)
        crop_frame = self.__apply_enhance(crop_frame)
        crop_frame = self.__normalize_crop_frame_enhance(crop_frame)
        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)
        paste_frame = self.__paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
        temp_frame = self.__blend_frame(temp_frame, paste_frame)

        return temp_frame

    ##################################
    #####   AUXILIARY FUNCTIONS   #####
    ##################################

    def __warp_face_by_kps(self, temp_frame: np.ndarray, kps: np.ndarray, landmark: np.ndarray, crop_size: tuple) -> tuple:
        """
        Warps the face in the frame to align with the keypoints.
        
        Args:
            temp_frame (`np.ndarray`): The input frame.
            kps (`np.ndarray`): The keypoints of the face.
            landmark (`np.ndarray`): The landmark points.
            crop_size (`tuple`): The size to crop the face to.
        
        Returns:
            `tuple`: The cropped frame and the affine transformation matrix.
        """
        normed_template = landmark * crop_size
        affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        crop_frame = cv2.warpAffine(temp_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
        return crop_frame, affine_matrix

    def __normalize_crop_frame_enhance(self, crop_frame: np.ndarray) -> np.ndarray:
        """
        Normalizes the enhanced cropped frame.
        
        Args:
            crop_frame (`np.ndarray`): The enhanced cropped frame.
        
        Returns:
            `np.ndarray`: The normalized cropped frame.
        """
        crop_frame = np.clip(crop_frame, -1, 1)
        crop_frame = (crop_frame + 1) / 2
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = (crop_frame * 255.0).round()
        crop_frame = crop_frame.astype(np.uint8)[:, :, ::-1]
        return crop_frame

    def __prepare_crop_frame_enhance(self, crop_frame: np.ndarray) -> np.ndarray:
        """
        Prepares the cropped frame for enhancement.
        
        Args:
            crop_frame (`np.ndarray`): The cropped frame.
        
        Returns:
            `np.ndarray`: The prepared cropped frame.
        """
        crop_frame = crop_frame[:, :, ::-1] / 255.0
        crop_frame = (crop_frame - 0.5) / 0.5
        crop_frame = np.expand_dims(crop_frame.transpose(2, 0, 1), axis=0).astype(np.float32)
        return crop_frame

    def __blend_frame(self, temp_frame: np.ndarray, paste_frame: np.ndarray) -> np.ndarray:
        """
        Blends the original frame with the enhanced frame.
        
        Args:
            temp_frame (`np.ndarray`): The original frame.
            paste_frame (`np.ndarray`): The enhanced frame to be blended.
        
        Returns:
            `np.ndarray`: The blended frame.
        """
        face_enhancer_blend = 1 - (100 / 100)
        temp_frame = cv2.addWeighted(temp_frame, face_enhancer_blend, paste_frame, 1 - face_enhancer_blend, 0)
        return temp_frame

    def __apply_enhance(self, crop_frame: np.ndarray) -> np.ndarray:
        """
        Applies the enhancement model to the cropped frame.
        
        Args:
            crop_frame (`np.ndarray`): The prepared cropped frame.
        
        Returns:
            `np.ndarray`: The enhanced cropped frame.
        """
        frame_processor = self.enhancer

        frame_processor_inputs = {}

        for frame_processor_input in frame_processor.get_inputs():
            if frame_processor_input.name == 'input':
                frame_processor_inputs[frame_processor_input.name] = crop_frame
            if frame_processor_input.name == 'weight':
                weight = np.array([1], dtype=np.double)
                frame_processor_inputs[frame_processor_input.name] = weight

        crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]

        return crop_frame

    def __paste_back(self, temp_frame: np.ndarray, crop_frame: np.ndarray, crop_mask: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Pastes the enhanced cropped frame back onto the original frame.
        
        Args:
            temp_frame (`np.ndarray`): The original frame.
            crop_frame (`np.ndarray`): The enhanced cropped frame.
            crop_mask (`np.ndarray`): The mask for the cropped frame.
            affine_matrix (`np.ndarray`): The affine transformation matrix.
        
        Returns:
            `np.ndarray`: The frame with the enhanced face pasted back.
        """
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_frame_size = temp_frame.shape[:2][::-1]
        inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
        inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode=cv2.BORDER_REPLICATE)

        if self.enhancer.get_providers() == 'CUDAExecutionProvider':
            temp_frame = cp.array(temp_frame)
            inverse_crop_mask = cp.array(inverse_crop_mask)
            inverse_crop_frame = cp.array(inverse_crop_frame)
        else:
            temp_frame = np.array(temp_frame)
            inverse_crop_mask = np.array(inverse_crop_mask)
            inverse_crop_frame = np.array(inverse_crop_frame)

        temp_frame[:, :, 0] = inverse_crop_mask * inverse_crop_frame[:, :, 0] + (1 - inverse_crop_mask) * temp_frame[:, :, 0]
        temp_frame[:, :, 1] = inverse_crop_mask * inverse_crop_frame[:, :, 1] + (1 - inverse_crop_mask) * temp_frame[:, :, 1]
        temp_frame[:, :, 2] = inverse_crop_mask * inverse_crop_frame[:, :, 2] + (1 - inverse_crop_mask) * temp_frame[:, :, 2]

        if self.enhancer.get_providers() == 'CUDAExecutionProvider':
            paste_frame = cp.asnumpy(temp_frame)
        else:
            paste_frame = temp_frame

        return paste_frame

    def __create_static_box_mask(self, crop_size: tuple, face_mask_blur: float, face_mask_padding: tuple) -> np.ndarray:
        """
        Creates a static box mask for the cropped frame.
        
        Args:
            crop_size (`tuple`): The size of the cropped frame.
            face_mask_blur (`float`): The amount of blur for the face mask.
            face_mask_padding (`tuple`): The padding for the face mask.
        
        Returns:
            `np.ndarray`: The static box mask.
        """
        blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
        blur_area = max(blur_amount // 2, 1)
        box_mask = np.ones(crop_size, np.float32)
        box_mask[:max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
        box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)):, :] = 0
        box_mask[:, :max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
        box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)):] = 0
        if blur_amount > 0:
            box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
        return box_mask
