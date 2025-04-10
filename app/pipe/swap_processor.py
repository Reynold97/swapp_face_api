import cv2
import numpy as np
from ray import serve
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union, Any

from app.utils.utils import Face


class SwapMode(str, Enum):
    """Enum defining the available face swap modes."""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    SORTED = "sorted"
    SIMILARITY = "similarity"


class SwapProcessorException(Exception):
    """Base exception class for SwapProcessor errors."""
    pass


class NoSourceFaceError(SwapProcessorException):
    """Exception raised when no face is detected in the source image."""
    def __init__(self, message="No face detected in the source image"):
        self.message = message
        super().__init__(self.message)


class NoTargetFaceError(SwapProcessorException):
    """Exception raised when no face is detected in the target image."""
    def __init__(self, message="No face detected in the target image"):
        self.message = message
        super().__init__(self.message)


class FaceModificationTracker:
    """
    Utility class for tracking face modifications and handling overlap issues
    when applying multiple face swaps to a single image.
    """
    
    def __init__(self, frame: np.ndarray):
        """
        Initialize the tracker with the original frame.
        
        Args:
            frame: The original image frame before any modifications
        """
        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        self.height, self.width = frame.shape[:2]
        
        # Mask to track which pixels have been modified (0 = unmodified, 1 = modified)
        self.modification_mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Store face regions for potential conflict resolution
        self.modified_regions = []  # List of (face, mask) tuples
    
    def register_face(self, face: Face) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a face region to be tracked.
        
        Args:
            face: The Face object containing keypoints
            
        Returns:
            Tuple containing:
                - The bounding box of the face [x1, y1, x2, y2]
                - The face mask for blending
        """
        # Create a face mask based on the facial keypoints
        face_mask = self._create_face_mask(face)
        
        # Get the face bounding box
        bbox = self._get_face_bbox(face)
        
        return bbox, face_mask
    
    def update_with_swap(self, swapped_region: np.ndarray, face: Face) -> np.ndarray:
        """
        Update the current frame with a newly swapped face region, handling potential conflicts.
        
        Args:
            swapped_region: The newly swapped face region
            face: The Face object corresponding to the swapped region
            
        Returns:
            The updated frame with the new swap integrated
        """
        # Get the face mask and bounding box
        bbox, face_mask = self.register_face(face)
        x1, y1, x2, y2 = bbox
        
        # Extract the region of interest from the current frame
        roi = self.current_frame[y1:y2, x1:x2].copy()
        
        # Extract the mask region
        mask_roi = face_mask[y1:y2, x1:x2]
        
        # Extract the modification mask region
        mod_mask_roi = self.modification_mask[y1:y2, x1:x2]
        
        # Calculate the overlap mask (where the current face overlaps with previous modifications)
        overlap_mask = np.minimum(mask_roi, mod_mask_roi)
        
        # Calculate the non-overlap mask (where the current face doesn't overlap)
        non_overlap_mask = np.maximum(0, mask_roi - overlap_mask)
        
        # Blend the swapped region with the current frame
        # For non-overlapping areas, use the swapped region
        # For overlapping areas, blend based on the modification mask
        blended_roi = roi.copy()
        swapped_region_roi = swapped_region[y1:y2, x1:x2]
        
        # Blend channels separately
        for c in range(3):  # RGB channels
            blended_roi[:, :, c] = (
                roi[:, :, c] * (1 - non_overlap_mask) +  # Keep original for non-mask areas
                swapped_region_roi[:, :, c] * non_overlap_mask  # Use swapped face for mask areas
            )
            
            if np.any(overlap_mask > 0):
                # For overlapping areas, use a weighted blend based on modification mask values
                blend_weight = overlap_mask / (mod_mask_roi + 0.001)  # Avoid division by zero
                blended_roi[:, :, c] = (
                    blended_roi[:, :, c] * (1 - blend_weight) +
                    swapped_region_roi[:, :, c] * blend_weight
                )
        
        # Update the current frame with the blended region
        result_frame = self.current_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_roi
        
        # Update the modification mask with the new face mask
        self.modification_mask[y1:y2, x1:x2] = np.maximum(mod_mask_roi, mask_roi)
        
        # Store the modified region
        self.modified_regions.append((face, face_mask))
        
        # Update the current frame
        self.current_frame = result_frame
        
        return result_frame
    
    def get_current_frame(self) -> np.ndarray:
        """
        Get the current frame with all modifications applied.
        
        Returns:
            The current frame
        """
        return self.current_frame
    
    def _create_face_mask(self, face: Face) -> np.ndarray:
        """
        Create a mask for the face based on its keypoints.
        
        Args:
            face: The Face object containing keypoints
            
        Returns:
            A float32 mask where 1.0 indicates the face region and 0.0 indicates non-face regions
        """
        # Initialize an empty mask
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Create a convex hull from the face keypoints
        kps = face.kps.astype(np.int32)
        hull = cv2.convexHull(kps)
        
        # Fill the convex hull with 1.0
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # Apply Gaussian blur to soften the mask edges for better blending
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        
        # Clip mask values to [0, 1] range
        return np.clip(mask, 0, 1)
    
    def _get_face_bbox(self, face: Face) -> Tuple[int, int, int, int]:
        """
        Get the bounding box coordinates for a face.
        
        Args:
            face: The Face object containing keypoints
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates
        """
        # Get the min/max x and y coordinates of the face keypoints
        kps = face.kps.astype(np.int32)
        x1 = max(0, np.min(kps[:, 0]) - 20)  # Add some margin
        y1 = max(0, np.min(kps[:, 1]) - 20)
        x2 = min(self.width, np.max(kps[:, 0]) + 20)
        y2 = min(self.height, np.max(kps[:, 1]) + 20)
        
        return int(x1), int(y1), int(x2), int(y2)


@serve.deployment()
class SwapProcessor:
    """
    A component for processing face swaps in various modes.
    
    This processor supports four modes:
    1. One to One: Apply one source face to one target face (standard swap)
    2. One to Many: Apply one source face to all target faces
    3. Sorted: Apply faces in spatial order (left-to-right or right-to-left)
    4. Similarity: Match source faces to target faces by similarity
    """

    def __init__(self, analyzer_handle, swapper_handle, enhancer_handle=None, codeformer_handle=None):
        """
        Initialize the SwapProcessor.
        
        Args:
            analyzer_handle: Handle to the FaceAnalyzer deployment
            swapper_handle: Handle to the FaceSwapper deployment
            enhancer_handle: Handle to the FaceEnhancer deployment (optional)
            codeformer_handle: Handle to the CodeFormerEnhancer deployment (optional)
        """
        self.analyzer_handle = analyzer_handle
        self.swapper_handle = swapper_handle
        self.enhancer_handle = enhancer_handle
        self.codeformer_handle = codeformer_handle

    async def process_swap(self, 
                          mode: str, 
                          source_frame: np.ndarray, 
                          target_frame: np.ndarray, 
                          direction: str = 'left_to_right',
                          enhance: bool = True,
                          use_codeformer: bool = False,
                          codeformer_fidelity: float = 0.5,
                          background_enhance: bool = True,
                          face_upsample: bool = True,
                          upscale: int = 2) -> np.ndarray:
        """
        Main entry point for processing face swaps with multiple modes.
        
        Args:
            mode: The swap mode ('one_to_one', 'one_to_many', 'sorted', or 'similarity')
            source_frame: The source image containing faces to use for swapping
            target_frame: The target image where faces will be swapped
            direction: The direction for sorted mode ('left_to_right' or 'right_to_left')
            enhance: Whether to enhance the result after swapping
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The processed image with swapped faces
            
        Raises:
            NoSourceFaceError: If no face is detected in the source image
            NoTargetFaceError: If no face is detected in the target image
            ValueError: If the swap mode is invalid
        """
        try:
            # Convert string mode to enum for easier handling
            swap_mode = SwapMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid swap mode: {mode}. Valid modes are: {[m.value for m in SwapMode]}")
        
        # Process based on mode
        if swap_mode == SwapMode.ONE_TO_ONE:
            return await self.one_to_one_swap(
                source_frame, target_frame, enhance, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
        elif swap_mode == SwapMode.ONE_TO_MANY:
            return await self.one_to_many_swap(
                source_frame, target_frame, enhance, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
        elif swap_mode == SwapMode.SORTED:
            return await self.sorted_swap(
                source_frame, target_frame, direction, enhance, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
        elif swap_mode == SwapMode.SIMILARITY:
            return await self.similarity_swap(
                source_frame, target_frame, enhance, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
        else:
            raise ValueError(f"Unsupported swap mode: {mode}")

    async def one_to_one_swap(self, 
                            source_frame: np.ndarray, 
                            target_frame: np.ndarray,
                            enhance: bool = True,
                            use_codeformer: bool = False,
                            codeformer_fidelity: float = 0.5,
                            background_enhance: bool = True,
                            face_upsample: bool = True,
                            upscale: int = 2) -> np.ndarray:
        """
        Apply one source face to one target face (standard swap).
        
        Args:
            source_frame: The source image containing the face to use for swapping
            target_frame: The target image where the face will be swapped
            enhance: Whether to enhance the result after swapping
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The processed image with swapped face
            
        Raises:
            NoSourceFaceError: If no face is detected in the source image
            NoTargetFaceError: If no face is detected in the target image
        """
        # Extract the source face (first face found)
        source_face = await self.analyzer_handle.extract_faces.remote(source_frame, index=0)
        if source_face is None:
            raise NoSourceFaceError()
            
        # Extract the target face (first face found)
        target_face = await self.analyzer_handle.extract_faces.remote(target_frame, index=0)
        if target_face is None:
            raise NoTargetFaceError()
            
        # Apply the swap
        result_frame = await self.swapper_handle.swap_face.remote(
            source_face, target_face, target_frame
        )
        
        # Enhance if requested
        if enhance:
            result_frame = await self._enhance_result(
                result_frame, [target_face], use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
            
        return result_frame

    async def one_to_many_swap(self, 
                             source_frame: np.ndarray, 
                             target_frame: np.ndarray,
                             enhance: bool = True,
                             use_codeformer: bool = False,
                             codeformer_fidelity: float = 0.5,
                             background_enhance: bool = True,
                             face_upsample: bool = True,
                             upscale: int = 2) -> np.ndarray:
        """
        Apply one source face to all detected faces in the target image.
        
        Args:
            source_frame: The source image containing the face to use for swapping
            target_frame: The target image where faces will be swapped
            enhance: Whether to enhance the result after swapping
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The processed image with swapped faces
            
        Raises:
            NoSourceFaceError: If no face is detected in the source image
            NoTargetFaceError: If no faces are detected in the target image
        """
        # Extract the source face (first face found)
        source_face = await self.analyzer_handle.extract_faces.remote(source_frame, index=0)
        if source_face is None:
            raise NoSourceFaceError()
            
        # Extract all target faces
        target_faces = await self.analyzer_handle.extract_faces.remote(target_frame, index=-1)
        if target_faces is None or len(target_faces) == 0:
            raise NoTargetFaceError("No faces detected in the target image")
            
        # Create a face modification tracker
        tracker = FaceModificationTracker(target_frame)
        
        # Apply the source face to each target face
        for target_face in target_faces:
            swapped_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, target_frame
            )
            tracker.update_with_swap(swapped_frame, target_face)
        
        # Get the final result
        result_frame = tracker.get_current_frame()
        
        # Enhance the final result if requested
        if enhance:
            result_frame = await self._enhance_result(
                result_frame, target_faces, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
            
        return result_frame

    async def sorted_swap(self, 
                        source_frame: np.ndarray, 
                        target_frame: np.ndarray,
                        direction: str = 'left_to_right',
                        enhance: bool = True,
                        use_codeformer: bool = False,
                        codeformer_fidelity: float = 0.5,
                        background_enhance: bool = True,
                        face_upsample: bool = True,
                        upscale: int = 2) -> np.ndarray:
        """
        Apply faces from source to target in a spatial order (left-to-right or right-to-left).
        
        Args:
            source_frame: The source image containing faces to use for swapping
            target_frame: The target image where faces will be swapped
            direction: The direction for sorting ('left_to_right' or 'right_to_left')
            enhance: Whether to enhance the result after swapping
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The processed image with swapped faces
            
        Raises:
            NoSourceFaceError: If no faces are detected in the source image
            NoTargetFaceError: If no faces are detected in the target image
        """
        # Extract all source faces
        source_faces = await self.analyzer_handle.extract_faces.remote(source_frame, index=-1)
        if source_faces is None or len(source_faces) == 0:
            raise NoSourceFaceError("No faces detected in the source image")
            
        # Extract all target faces
        target_faces = await self.analyzer_handle.extract_faces.remote(target_frame, index=-1)
        if target_faces is None or len(target_faces) == 0:
            raise NoTargetFaceError("No faces detected in the target image")
            
        # Sort source and target faces by position
        sorted_source_faces = await self.analyzer_handle.sort_faces_by_position.remote(source_faces, direction)
        sorted_target_faces = await self.analyzer_handle.sort_faces_by_position.remote(target_faces, direction)
        
        # Create a face modification tracker
        tracker = FaceModificationTracker(target_frame)
        
        # Determine how many faces to swap (minimum of source and target faces)
        num_faces_to_swap = min(len(sorted_source_faces), len(sorted_target_faces))
        
        # Apply the swaps in order
        for i in range(num_faces_to_swap):
            source_face = sorted_source_faces[i]
            target_face = sorted_target_faces[i]
            
            swapped_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, target_frame
            )
            tracker.update_with_swap(swapped_frame, target_face)
        
        # Get the final result
        result_frame = tracker.get_current_frame()
        
        # Enhance the final result if requested
        if enhance:
            # We only enhance the faces that were swapped
            swapped_target_faces = sorted_target_faces[:num_faces_to_swap]
            result_frame = await self._enhance_result(
                result_frame, swapped_target_faces, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
            
        return result_frame

    async def similarity_swap(self, 
                            source_frame: np.ndarray, 
                            target_frame: np.ndarray,
                            enhance: bool = True,
                            use_codeformer: bool = False,
                            codeformer_fidelity: float = 0.5,
                            background_enhance: bool = True,
                            face_upsample: bool = True,
                            upscale: int = 2) -> np.ndarray:
        """
        Apply faces from source to target based on embedding similarity.
        
        Args:
            source_frame: The source image containing faces to use for swapping
            target_frame: The target image where faces will be swapped
            enhance: Whether to enhance the result after swapping
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The processed image with swapped faces
            
        Raises:
            NoSourceFaceError: If no faces are detected in the source image
            NoTargetFaceError: If no faces are detected in the target image
        """
        # Extract all source faces
        source_faces = await self.analyzer_handle.extract_faces.remote(source_frame, index=-1)
        if source_faces is None or len(source_faces) == 0:
            raise NoSourceFaceError("No faces detected in the source image")
            
        # Extract all target faces
        target_faces = await self.analyzer_handle.extract_faces.remote(target_frame, index=-1)
        if target_faces is None or len(target_faces) == 0:
            raise NoTargetFaceError("No faces detected in the target image")
            
        # Find optimal face matches based on similarity
        face_matches = await self.analyzer_handle.match_faces_greedy.remote(source_faces, target_faces)
        
        # Create a face modification tracker
        tracker = FaceModificationTracker(target_frame)
        
        # Track which target faces have been swapped for enhancement
        swapped_target_faces = []
        
        # Apply the swaps based on similarity matches
        for source_idx, target_idx, _ in face_matches:
            source_face = source_faces[source_idx]
            target_face = target_faces[target_idx]
            
            # Store swapped face for enhancement
            swapped_target_faces.append(target_face)
            
            # Apply the swap
            swapped_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, target_frame
            )
            tracker.update_with_swap(swapped_frame, target_face)
        
        # Get the final result
        result_frame = tracker.get_current_frame()
        
        # Enhance the final result if requested
        if enhance:
            result_frame = await self._enhance_result(
                result_frame, swapped_target_faces, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale
            )
            
        return result_frame

    async def _enhance_result(self, 
                            frame: np.ndarray, 
                            faces: List[Face],
                            use_codeformer: bool = False,
                            codeformer_fidelity: float = 0.5,
                            background_enhance: bool = True,
                            face_upsample: bool = True,
                            upscale: int = 2) -> np.ndarray:
        """
        Helper method to enhance the result after all swaps are completed.
        
        Args:
            frame: The frame with swapped faces to enhance
            faces: List of Face objects that were swapped and need enhancement
            use_codeformer: Whether to use CodeFormer for enhancement
            codeformer_fidelity: Balance between quality and fidelity (0.0-1.0) for CodeFormer
            background_enhance: Whether to enhance the background (CodeFormer only)
            face_upsample: Whether to upsample the faces (CodeFormer only)
            upscale: The upscale factor for enhancing (CodeFormer only)
            
        Returns:
            The enhanced frame
        """
        # Make sure we have an enhancer available
        if not self.enhancer_handle and not self.codeformer_handle:
            return frame
            
        enhanced_frame = frame.copy()
        
        # Choose which enhancer to use
        if use_codeformer and self.codeformer_handle:
            # For CodeFormer, we can enhance each face individually
            for face in faces:
                enhanced_frame = await self.codeformer_handle.enhance_face.remote(
                    face, 
                    enhanced_frame, 
                    codeformer_fidelity,
                    background_enhance,
                    face_upsample,
                    upscale
                )
        elif self.enhancer_handle:
            # For the standard enhancer, enhance each face
            for face in faces:
                enhanced_frame = await self.enhancer_handle.enhance_face.remote(
                    face, enhanced_frame
                )
                
        return enhanced_frame