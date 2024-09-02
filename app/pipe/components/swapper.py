import cv2
import onnx
import cupy as cp
import numpy as np
import onnxruntime
from ray import serve
from onnx import numpy_helper
from app.utils.utils import Face, conditional_download


################################
#########   SWAPPER   ##########
################################

@serve.deployment()
class FaceSwapper:
    """
    A class used to swap faces in a given frame using `inswapper_128` model.
    """

    def __init__(self):
        """
        Initializes the FaceSwapper class by downloading and setting up the ONNX model for face swapping.
        """
        self.swapper = onnxruntime.InferenceSession(
            conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/inswapper_128_fp16.onnx'),
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
        )
        self.model_matrix = numpy_helper.to_array(onnx.load(
            conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/inswapper_128_fp16.onnx')
        ).graph.initializer[-1])
        
        print(f'SWAPPER is using the following provider(s): {self.swapper.get_providers()}')

    def swap_face(self, source_face: Face, target_face: Face, temp_frame: np.ndarray) -> np.ndarray:
        """
        Swaps the source face with the target face in the given frame and returns the resulting frame.
        
        Args:
            source_face (`Face`): The source face namedtuple containing keypoints and embedding.
            target_face (`Face`): The target face namedtuple containing keypoints and embedding.
            temp_frame (`np.ndarray`): The input image frame.
        
        Returns:
            `np.ndarray`: The frame with the face swapped.
        """
        if source_face is None or target_face is None:
            return None

        arcface_128_v2_landmark = np.array([
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453]
        ])

        model_size = (128, 128)
        crop_frame, affine_matrix = self.__warp_face_by_kps(temp_frame, target_face.kps, arcface_128_v2_landmark, model_size)
        crop_mask_list = [self.__create_static_box_mask(crop_frame.shape[:2][::-1], 0.3, (0, 0, 0, 0))]
        crop_frame = self.__prepare_crop_frame_swap(crop_frame)
        crop_frame = self.__apply_swap(source_face, crop_frame)
        crop_frame = self.__normalize_crop_frame_swap(crop_frame)
        crop_mask = np.minimum.reduce(crop_mask_list).clip(0, 1)

        temp_frame = self.__paste_back(temp_frame, crop_frame, crop_mask, affine_matrix)
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

    def __prepare_source_embedding(self, source_face: Face) -> np.ndarray:
        """
        Prepares the source face embedding.
        
        Args:
            source_face (`Face`): The source face namedtuple containing keypoints and embedding.
        
        Returns:
            `np.ndarray`: The prepared source embedding.
        """
        model_matrix = self.model_matrix
        source_embedding = source_face.embedding.reshape((1, -1))
        source_embedding = np.dot(source_embedding, model_matrix) / np.linalg.norm(source_embedding)
        return source_embedding

    def __apply_swap(self, source_face: Face, crop_frame: np.ndarray) -> np.ndarray:
        """
        Applies the face swap using the ONNX model.
        
        Args:
            source_face (`Face`): The source face namedtuple containing keypoints and embedding.
            crop_frame (`np.ndarray`): The prepared cropped frame.
        
        Returns:
            `np.ndarray`: The swapped face frame.
        """
        frame_processor = self.swapper

        frame_processor_inputs = {}
        for frame_processor_input in frame_processor.get_inputs():
            if frame_processor_input.name == 'source':
                frame_processor_inputs[frame_processor_input.name] = self.__prepare_source_embedding(source_face)
            if frame_processor_input.name == 'target':
                frame_processor_inputs[frame_processor_input.name] = crop_frame

        crop_frame = frame_processor.run(None, frame_processor_inputs)[0][0]

        return crop_frame

    def __paste_back(self, temp_frame: np.ndarray, crop_frame: np.ndarray, crop_mask: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
        """
        Pastes the swapped face frame back onto the original frame.
        
        Args:
            temp_frame (`np.ndarray`): The original frame.
            crop_frame (`np.ndarray`): The swapped face frame.
            crop_mask (`np.ndarray`): The mask for the cropped frame.
            affine_matrix (`np.ndarray`): The affine transformation matrix.
        
        Returns:
            `np.ndarray`: The frame with the swapped face pasted back.
        """
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        temp_frame_size = temp_frame.shape[:2][::-1]
        inverse_crop_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_frame_size).clip(0, 1)
        inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode=cv2.BORDER_REPLICATE)

        if self.swapper.get_providers() == 'CUDAExecutionProvider':
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

        if self.swapper.get_providers() == 'CUDAExecutionProvider':
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

    def __normalize_crop_frame_swap(self, crop_frame: np.ndarray) -> np.ndarray:
        """
        Normalizes the swapped cropped frame.
        
        Args:
            crop_frame (`np.ndarray`): The swapped cropped frame.
        
        Returns:
            `np.ndarray`: The normalized cropped frame.
        """
        crop_frame = crop_frame.transpose(1, 2, 0)
        crop_frame = (crop_frame * 255.0).round()
        crop_frame = crop_frame[:, :, ::-1]
        return crop_frame

    def __prepare_crop_frame_swap(self, crop_frame: np.ndarray) -> np.ndarray:
        """
        Prepares the cropped frame for swapping.
        
        Args:
            crop_frame (`np.ndarray`): The cropped frame.
        
        Returns:
            `np.ndarray`: The prepared cropped frame.
        """
        model_mean = [0.0, 0.0, 0.0]
        model_standard_deviation = [1.0, 1.0, 1.0]
        crop_frame = crop_frame[:, :, ::-1] / 255.0
        crop_frame = (crop_frame - model_mean) / model_standard_deviation
        crop_frame = crop_frame.transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis=0).astype(np.float32)
        return crop_frame
