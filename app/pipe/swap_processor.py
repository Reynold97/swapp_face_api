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
                          upscale: int = 2,
                          face_refinement_steps: int = 1) -> np.ndarray:
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
        # Validate the refinement steps
        face_refinement_steps = max(1, min(face_refinement_steps, 5))

        try:
            # Convert string mode to enum for easier handling
            swap_mode = SwapMode(mode.lower())
        except ValueError:
            raise ValueError(f"Invalid swap mode: {mode}. Valid modes are: {[m.value for m in SwapMode]}")
        
        # Process based on mode
        if swap_mode == SwapMode.ONE_TO_ONE:
            return await self.one_to_one_swap(
                source_frame, target_frame, enhance, use_codeformer,
                codeformer_fidelity, background_enhance, face_upsample, upscale, 
                face_refinement_steps
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
                            upscale: int = 2,
                            face_refinement_steps: int = 1) -> np.ndarray:
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
            face_refinement_steps: Number of iterative swaps
            
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
            
        # Initialize result with the target frame
        result_frame = target_frame.copy()
        current_target_face = target_face
        
        # Apply the swap multiple times 
        for i in range(face_refinement_steps):
            result_frame = await self.swapper_handle.swap_face.remote(
                source_face, current_target_face, result_frame
            )
            
            # If we need to do more iterations, re-detect the face
            if i < face_refinement_steps - 1:
                current_target_face = await self.analyzer_handle.extract_faces.remote(result_frame, index=0)
                if current_target_face is None:
                    # If face detection fails, stop iterating
                    break
        
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
        """
        # Extract the source face (first face found)
        source_face = await self.analyzer_handle.extract_faces.remote(source_frame, index=0)
        if source_face is None:
            raise NoSourceFaceError()
            
        # Extract all target faces
        target_faces = await self.analyzer_handle.extract_faces.remote(target_frame, index=-1)
        if target_faces is None or len(target_faces) == 0:
            raise NoTargetFaceError("No faces detected in the target image")
        
        # Apply swaps sequentially
        result_frame = target_frame.copy()
        
        for target_face in target_faces:
            result_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, result_frame
            )
        
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
        The direction controls how target faces are selected, but source faces are always
        anchored in left-to-right order.
        """
        # Extract all source faces
        source_faces = await self.analyzer_handle.extract_faces.remote(source_frame, index=-1)
        if source_faces is None or len(source_faces) == 0:
            raise NoSourceFaceError("No faces detected in the source image")
            
        # Extract all target faces
        target_faces = await self.analyzer_handle.extract_faces.remote(target_frame, index=-1)
        if target_faces is None or len(target_faces) == 0:
            raise NoTargetFaceError("No faces detected in the target image")
        
        # Helper function to sort faces by x-position
        def sort_faces_by_position(faces, sort_direction):
            face_positions = []
            for face in faces:
                centroid_x = np.mean(face.kps[:, 0])
                face_positions.append((face, centroid_x))
            
            sorted_faces = sorted(face_positions, key=lambda x: x[1])
            
            if sort_direction.lower() == 'right_to_left':
                sorted_faces.reverse()
                
            return [face for face, _ in sorted_faces]
        
        # ALWAYS sort source faces left-to-right (anchor them in a fixed order)
        sorted_source_faces = sort_faces_by_position(source_faces, 'left_to_right')
        
        # Sort target faces according to the requested direction
        sorted_target_faces = sort_faces_by_position(target_faces, direction)
        
        print(f"Source faces anchored left-to-right, target faces sorted {direction}")
        
        # Apply swaps sequentially
        result_frame = target_frame.copy()
        
        # Determine how many faces to swap (minimum of source and target faces)
        num_faces_to_swap = min(len(sorted_source_faces), len(sorted_target_faces))
        
        # Apply the swaps in order
        for i in range(num_faces_to_swap):
            source_face = sorted_source_faces[i]
            target_face = sorted_target_faces[i]
            
            source_x = np.mean(source_face.kps[:, 0])
            target_x = np.mean(target_face.kps[:, 0]) 
            print(f"Swapping source face at x:{source_x:.1f} to target face at x:{target_x:.1f}")
            
            result_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, result_frame
            )
        
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
        
        # Apply swaps sequentially
        result_frame = target_frame.copy()
        
        # Track which target faces have been swapped for enhancement
        swapped_target_faces = []
        
        # Apply the swaps based on similarity matches
        for source_idx, target_idx, _ in face_matches:
            source_face = source_faces[source_idx]
            target_face = target_faces[target_idx]
            
            # Apply the swap
            result_frame = await self.swapper_handle.swap_face.remote(
                source_face, target_face, result_frame
            )
            
            # Store swapped face for enhancement
            swapped_target_faces.append(target_face)
        
        # Enhance the final result if requested
        if enhance and swapped_target_faces:
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