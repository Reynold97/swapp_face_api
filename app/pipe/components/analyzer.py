import cv2
import numpy as np
import onnxruntime
from ray import serve
from app.utils.utils import Face, conditional_download


################################
#########   ANALYZER   #########
################################

@serve.deployment()
class FaceAnalyzer:
    """
    A class used to analyze faces in a given frame using models for face detection and recognition.
    """

    def __init__(self):
        """
        Initializes the FaceAnalyzer class by downloading and setting up the ONNX models for face detection and recognition.
        """
        self.recognizer = onnxruntime.InferenceSession(
            conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/arcface_w600k_r50_fp16.onnx'),
            providers=['CUDAExecutionProvider']
        )
        self.detector = onnxruntime.InferenceSession(
            conditional_download('https://huggingface.co/leandro-driguez/swap-faces/resolve/main/retinaface_10g_fp16.onnx'),
            providers=['CUDAExecutionProvider']
        )

    def extract_faces(self, frame: np.ndarray, index: int = 0):
        """
        Extracts faces from a given frame and returns the face at the specified index.
        
        Args:
            frame (`np.ndarray`): The input image frame.
            index (`int`, optional): The index of the face to return. Defaults to `0`. If negative, returns all faces.
        
        Returns:
            `Face`: A Face namedtuple containing keypoints and embedding of the face at the specified index.
        """
        if frame is None:
            return None

        face_detector_width, face_detector_height = 640, 640
        frame_height, frame_width, _ = frame.shape
        temp_frame = self.__resize_frame_resolution(frame, face_detector_width, face_detector_height)
        temp_frame_height, temp_frame_width, _ = temp_frame.shape
        ratio_height = frame_height / temp_frame_height
        ratio_width = frame_width / temp_frame_width

        bbox_list, kps_list, score_list = self.__detect_with_retinaface(
            temp_frame, temp_frame_height, temp_frame_width, 
            face_detector_height, face_detector_width, 
            ratio_height, ratio_width
        )

        faces = self.__create_faces(frame, bbox_list, kps_list, score_list)

        if len(faces) == 0:
            return None

        if index < 0:
            return faces
        return faces[index]

    ###################################
    #####   AUXILIARY FUNCTIONS   #####
    ###################################

    def __create_static_anchors(self, feature_stride: int, anchor_total: int, stride_height: int, stride_width: int) -> np.ndarray:
        """
        Creates static anchors for the feature map.
        
        Args:
            feature_stride (`int`): The stride size of the feature map.
            anchor_total (`int`): The total number of anchors.
            stride_height (`int`): The height of the feature map.
            stride_width (`int`): The width of the feature map.
        
        Returns:
            `np.ndarray`: An array of anchor points.
        """
        y, x = np.mgrid[:stride_height, :stride_width][::-1]
        anchors = np.stack((y, x), axis=-1)
        anchors = (anchors * feature_stride).reshape((-1, 2))
        anchors = np.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
        return anchors

    def __distance_to_bbox(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """
        Converts distance to bounding box coordinates.
        
        Args:
            points (`np.ndarray`): The center points.
            distance (`np.ndarray`): The distance to the bounding box edges.
        
        Returns:
            `np.ndarray`: The bounding box coordinates.
        """
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        bbox = np.column_stack([x1, y1, x2, y2])
        return bbox

    def __distance_to_kps(self, points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        """
        Converts distance to keypoints.
        
        Args:
            points (`np.ndarray`): The center points.
            distance (`np.ndarray`): The distance to the keypoints.
        
        Returns:
            `np.ndarray`: The keypoints.
        """
        x = points[:, 0::2] + distance[:, 0::2]
        y = points[:, 1::2] + distance[:, 1::2]
        kps = np.stack((x, y), axis=-1)
        return kps

    def __detect_with_retinaface(self, temp_frame: np.ndarray, temp_frame_height: int, temp_frame_width: int, 
                                 face_detector_height: int, face_detector_width: int, 
                                 ratio_height: float, ratio_width: float):
        """
        Detects faces in a given frame using the RetinaFace model.
        
        Args:
            temp_frame (`np.ndarray`): The preprocessed frame.
            temp_frame_height (`int`): The height of the preprocessed frame.
            temp_frame_width (`int`): The width of the preprocessed frame.
            face_detector_height (`int`): The height of the face detector.
            face_detector_width (`int`): The width of the face detector.
            ratio_height (`float`): The ratio of the original frame height to the preprocessed frame height.
            ratio_width (`float`): The ratio of the original frame width to the preprocessed frame width.
        
        Returns:
            `tuple`: A tuple containing the list of bounding boxes, keypoints, and scores.
        """
        face_detector = self.detector

        prepare_frame = np.zeros((face_detector_height, face_detector_width, 3))
        prepare_frame[:temp_frame_height, :temp_frame_width, :] = temp_frame
        temp_frame = (prepare_frame - 127.5) / 128.0
        temp_frame = np.expand_dims(temp_frame.transpose(2, 0, 1), axis=0).astype(np.float16)
        
        detections = face_detector.run(None, {face_detector.get_inputs()[0].name: temp_frame})
        
        bbox_list = []
        kps_list = []
        score_list = []
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 2
        for index, feature_stride in enumerate(feature_strides):
            keep_indices = np.where(detections[index] >= 0.5)[0]
            if keep_indices.any():
                stride_height = face_detector_height // feature_stride
                stride_width = face_detector_width // feature_stride
                anchors = self.__create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
                bbox_raw = detections[index + feature_map_channel] * feature_stride
                kps_raw = detections[index + feature_map_channel * 2] * feature_stride
                for bbox in self.__distance_to_bbox(anchors, bbox_raw)[keep_indices]:
                    bbox_list.append(np.array(
                    [
                        bbox[0] * ratio_width,
                        bbox[1] * ratio_height,
                        bbox[2] * ratio_width,
                        bbox[3] * ratio_height
                    ]))
                for kps in self.__distance_to_kps(anchors, kps_raw)[keep_indices]:
                    kps_list.append(kps * [ratio_width, ratio_height])
                for score in detections[index][keep_indices]:
                    score_list.append(score[0])
        return bbox_list, kps_list, score_list

    def __resize_frame_resolution(self, frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        """
        Resizes the frame to fit within the specified maximum width and height while maintaining the aspect ratio.
        
        Args:
            frame (`np.ndarray`): The input frame.
            max_width (`int`): The maximum width for resizing.
            max_height (`int`): The maximum height for resizing.
        
        Returns:
            `np.ndarray`: The resized frame.
        """
        height, width = frame.shape[:2]

        if height > max_height or width > max_width:
            scale = min(max_height / height, max_width / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    def __apply_nms(self, bbox_list: list, iou_threshold: float) -> list:
        """
        Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        
        Args:
            bbox_list (`list`): The list of bounding boxes.
            iou_threshold (`float`): The Intersection over Union (IoU) threshold for NMS.
        
        Returns:
            `list`: The list of indices of bounding boxes to keep.
        """
        keep_indices = []
        dimension_list = np.reshape(bbox_list, (-1, 4))
        x1 = dimension_list[:, 0]
        y1 = dimension_list[:, 1]
        x2 = dimension_list[:, 2]
        y2 = dimension_list[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.arange(len(bbox_list))
        while indices.size > 0:
            index = indices[0]
            remain_indices = indices[1:]
            keep_indices.append(index)
            xx1 = np.maximum(x1[index], x1[remain_indices])
            yy1 = np.maximum(y1[index], y1[remain_indices])
            xx2 = np.minimum(x2[index], x2[remain_indices])
            yy2 = np.minimum(y2[index], y2[remain_indices])
            width = np.maximum(0, xx2 - xx1 + 1)
            height = np.maximum(0, yy2 - yy1 + 1)
            iou = width * height / (areas[index] + areas[remain_indices] - width * height)
            indices = indices[np.where(iou <= iou_threshold)[0] + 1]
        return keep_indices

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

    def __calc_embedding(self, temp_frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """
        Calculates the embedding of the face using the ArcFace model.
        
        Args:
            temp_frame (`np.ndarray`): The input frame.
            kps (`np.ndarray`): The keypoints of the face.
        
        Returns:
            `np.ndarray`: The face embedding.
        """
        face_recognizer = self.recognizer
        
        arcface_112_v2 = np.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089]
        ])

        crop_frame, _ = self.__warp_face_by_kps(temp_frame, kps, arcface_112_v2, (112, 112))
        crop_frame = crop_frame.astype(np.float32) / 127.5 - 1
        crop_frame = crop_frame[:, :, ::-1].transpose(2, 0, 1)
        crop_frame = np.expand_dims(crop_frame, axis=0).astype(np.float16)

        embedding = face_recognizer.run(None, {face_recognizer.get_inputs()[0].name: crop_frame})[0]

        embedding = embedding.ravel()
        return embedding

    def __create_faces(self, frame: np.ndarray, bbox_list: list, kps_list: list, score_list: list) -> list:
        """
        Creates a list of Face namedtuples from the detected bounding boxes and keypoints.
        
        Args:
            frame (`np.ndarray`): The input frame.
            bbox_list (`list`): The list of bounding boxes.
            kps_list (`list`): The list of keypoints.
            score_list (`list`): The list of scores.
        
        Returns:
            `list`: A list of Face namedtuples.
        """
        faces = []

        sort_indices = np.argsort(-np.array(score_list))
        bbox_list = [bbox_list[index] for index in sort_indices]
        kps_list = [kps_list[index] for index in sort_indices]
        score_list = [score_list[index] for index in sort_indices]

        keep_indices = self.__apply_nms(bbox_list, 0.4)

        for index in keep_indices:
            kps = kps_list[index]
            embedding = self.__calc_embedding(frame, kps)
            faces.append(Face(kps=kps, embedding=embedding))
        return faces
