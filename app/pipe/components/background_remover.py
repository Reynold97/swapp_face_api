import os
import numpy as np
import cv2
import cupy as cp
import tensorrt as trt
from tensorrt_bindings import Logger
import time
from PIL import Image
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from ray import serve
import logging

from app.utils.tensorrt_utils import (
    allocate_buffers, free_buffers, do_inference, load_engine
)

logger = logging.getLogger("ray.serve")


@serve.deployment()
class BackgroundRemover:
    """
    A Ray Serve deployment for ultra-fast background removal using TensorRT FP16.
    """

    def __init__(self, optimal_resolution: int = 1024, max_batch_size: int = 20):
        """
        Initialize the BackgroundRemover with TensorRT engine.
        
        Args:
            optimal_resolution: Resolution for processing (default 1024)
            max_batch_size: Maximum batch size for inference (default 20)
        """
        self.optimal_resolution = optimal_resolution
        self.max_batch_size = max_batch_size
        self.trt_logger = Logger(Logger.INFO)
        
        # Construct engine path relative to project root
        engine_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', '..', 
            'models', 
            'engine_fp16.trt'
        )
        engine_path = os.path.abspath(engine_path)
        
        logger.info(f"Loading TensorRT engine from: {engine_path}")
        
        if not os.path.exists(engine_path):
            raise ValueError(f"TensorRT engine not found at: {engine_path}")
        
        self.engine, self.context = self._load_engine(engine_path)
        
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="batch_postprocess")
        self._lock = threading.Lock()
        
        # FP16-optimized GPU arrays
        self.mean_gpu = cp.array([0.485, 0.456, 0.406], dtype=cp.float16).reshape(3, 1, 1)
        self.std_gpu = cp.array([0.229, 0.224, 0.225], dtype=cp.float16).reshape(3, 1, 1)
        
        # FP16 workspace arrays
        self.img_workspace = cp.zeros((optimal_resolution, optimal_resolution, 3), dtype=cp.float16)
        self.img_normalized = cp.zeros((3, optimal_resolution, optimal_resolution), dtype=cp.float16)
        self.mask_workspace = cp.zeros((optimal_resolution, optimal_resolution), dtype=cp.float16)
        self.alpha_workspace = cp.zeros((optimal_resolution, optimal_resolution, 1), dtype=cp.float16)
        self.result_workspace = cp.zeros((optimal_resolution, optimal_resolution, 4), dtype=cp.float16)
        
        # Allocate TensorRT buffers
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        
        # Blur kernel for refinement
        self.blur_kernel_15 = cp.asarray(
            cv2.getGaussianKernel(15, 0) @ cv2.getGaussianKernel(15, 0).T, 
            dtype=cp.float16
        )
        
        logger.info("BackgroundRemover initialized successfully")

    def _load_engine(self, engine_path: str):
        """Load and deserialize TensorRT engine."""
        if not os.path.exists(engine_path):
            raise ValueError(f"Engine file does not exist: {engine_path}")
            
        runtime = trt.Runtime(self.trt_logger)
        engine_data = load_engine(engine_path)
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
            
        context = engine.create_execution_context()
        return engine, context
    
    def preprocess_ultra_fast(self, image: Image.Image, original_size: tuple) -> np.ndarray:
        """Preprocess image for TensorRT inference using FP16."""
        if image.size != (self.optimal_resolution, self.optimal_resolution):
            image = image.resize((self.optimal_resolution, self.optimal_resolution), Image.LANCZOS)
        
        # FP16 pipeline
        img_array = cp.asarray(np.array(image, dtype=np.uint8))
        self.img_workspace[:] = img_array.astype(cp.float16) / cp.float16(255.0)
        
        # FP16 normalization
        self.img_normalized[:] = cp.transpose(self.img_workspace, (2, 0, 1))
        self.img_normalized -= self.mean_gpu
        self.img_normalized /= self.std_gpu
        
        return cp.asnumpy(self.img_normalized.ravel())
    
    def predict_mask_gpu(self, img_data: np.ndarray) -> cp.ndarray:
        """Run TensorRT inference to predict alpha mask."""
        with self._lock:
            # Convert to FP16
            fp16_data = img_data.astype(np.float16)
            np.copyto(self.inputs[0].host, fp16_data)
            
            trt_outputs = do_inference(
                self.context, self.engine, self.bindings, 
                self.inputs, self.outputs, self.stream
            )
            
            # FP16 output processing with GPU sigmoid
            pred_flat = cp.asarray(trt_outputs[-1], dtype=cp.float16)
            pred_flat = cp.float16(1.0) / (cp.float16(1.0) + cp.exp(-pred_flat))
            self.mask_workspace[:] = pred_flat.reshape(self.optimal_resolution, self.optimal_resolution)
            return self.mask_workspace.copy()
    
    def apply_mask_ultra_fast(self, original: Image.Image, mask_gpu: cp.ndarray, original_size: tuple) -> Image.Image:
        """Apply GPU-accelerated alpha mask with refinement."""
        original_width, original_height = original_size
        
        processing_img = original
        if original.size != (self.optimal_resolution, self.optimal_resolution):
            processing_img = original.resize((self.optimal_resolution, self.optimal_resolution), Image.LANCZOS)
        
        # FP16 local workspaces
        img_workspace_local = cp.zeros((self.optimal_resolution, self.optimal_resolution, 3), dtype=cp.float16)
        alpha_workspace_local = cp.zeros((self.optimal_resolution, self.optimal_resolution, 1), dtype=cp.float16)
        result_workspace_local = cp.zeros((self.optimal_resolution, self.optimal_resolution, 4), dtype=cp.float16)
        
        # FP16 GPU conversion
        img_gpu = cp.asarray(np.array(processing_img.convert("RGB"), dtype=np.uint8), dtype=cp.float16) / 255.0
        
        alpha_workspace_local[:, :, 0] = cp.clip(mask_gpu, 0, 1)
        
        # Refinement with blur
        alpha_2d = alpha_workspace_local[:, :, 0]
        alpha_blurred = cp.asarray(
            cv2.GaussianBlur(cp.asnumpy(alpha_2d.astype(cp.float32)), (15, 15), 0), 
            dtype=cp.float16
        )
        alpha_blurred_3d = alpha_blurred[:, :, cp.newaxis] + cp.float16(1e-4)
        
        alpha_3d = alpha_workspace_local
        fg_alpha = img_gpu * alpha_3d
        bg_alpha = img_gpu * (1 - alpha_3d)
        
        # FP16 blur operations
        fg_cpu = cp.asnumpy(fg_alpha.astype(cp.float32))
        bg_cpu = cp.asnumpy(bg_alpha.astype(cp.float32))
        
        fg_blurred = cp.asarray(cv2.GaussianBlur(fg_cpu, (15, 15), 0), dtype=cp.float16)
        bg_blurred = cp.asarray(cv2.GaussianBlur(bg_cpu, (15, 15), 0), dtype=cp.float16)
        
        # FP16 composition
        fg_est = fg_blurred / alpha_blurred_3d
        bg_est = bg_blurred / (1 - alpha_blurred_3d + cp.float16(1e-4))
        
        refined = fg_est + alpha_3d * (img_gpu - alpha_3d * fg_est - (1 - alpha_3d) * bg_est)
        refined = cp.clip(refined, 0, 1)
        
        result_workspace_local[:, :, :3] = refined
        result_workspace_local[:, :, 3:] = alpha_workspace_local
        
        # Convert to CPU
        result_cpu = cp.asnumpy((result_workspace_local.astype(cp.float32) * 255).astype(cp.uint8))
        result_pil = Image.fromarray(result_cpu, "RGBA")
        
        if (original_width, original_height) != (self.optimal_resolution, self.optimal_resolution):
            result_pil = result_pil.resize((original_width, original_height), Image.LANCZOS)
        
        return result_pil
    
    def process_with_background_ultra_fast(self, foreground: Image.Image, background: Image.Image) -> Image.Image:
        """Ultra-fast GPU-based alpha blending."""
        fg_w, fg_h = foreground.size
        
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')
        
        background_resized = background.resize((fg_w, fg_h), Image.LANCZOS)
        
        # FP16 GPU compositing
        fg_array = np.array(foreground, dtype=np.float32) / 255.0
        bg_array = np.array(background_resized.convert("RGB"), dtype=np.float32) / 255.0
        
        fg_gpu = cp.asarray(fg_array, dtype=cp.float16)
        bg_gpu = cp.asarray(bg_array, dtype=cp.float16)
        
        alpha = fg_gpu[:, :, 3:4]
        one_minus_alpha = cp.float16(1.0) - alpha
        
        result_gpu = cp.zeros((fg_h, fg_w, 3), dtype=cp.float16)
        
        # FP16 alpha blending
        result_gpu[:, :, :3] = (fg_gpu[:, :, :3] * alpha + 
                               bg_gpu[:, :, :3] * one_minus_alpha)
        
        result_gpu = cp.clip(result_gpu, 0, 1)
        result_array = cp.asnumpy((result_gpu.astype(cp.float32) * 255).astype(cp.uint8))
        
        result_img = Image.fromarray(result_array, 'RGB')
        
        return result_img

    def remove_background_batch(self, input_images: List[Image.Image], 
                               background_images: Optional[List[Image.Image]] = None) -> List[Image.Image]:
        """
        Remove backgrounds from a batch of images.
        
        Args:
            input_images: List of PIL Images to process
            background_images: Optional list of background images to apply
            
        Returns:
            List of processed PIL Images
        """
        start_time = time.perf_counter()
        batch_size = len(input_images)
        
        if batch_size > self.max_batch_size:
            results = []
            for i in range(0, batch_size, self.max_batch_size):
                chunk_end = min(i + self.max_batch_size, batch_size)
                chunk_images = input_images[i:chunk_end]
                chunk_backgrounds = background_images[i:chunk_end] if background_images else None
                chunk_results = self.remove_background_batch(chunk_images, chunk_backgrounds)
                results.extend(chunk_results)
            return results
        
        original_sizes = [img.size for img in input_images]
        
        logger.info(f"Processing FP16 batch: {batch_size} images")
        
        preprocess_start = time.perf_counter()
        masks_gpu = []
        
        for i, img in enumerate(input_images):
            img_data = self.preprocess_ultra_fast(img, original_sizes[i])
            mask = self.predict_mask_gpu(img_data)
            masks_gpu.append(mask.copy())
        
        preprocess_inference_time = time.perf_counter() - preprocess_start
        
        postprocess_start = time.perf_counter()
        
        def process_single_item(args):
            i, img, mask, original_size = args
            bg_img = background_images[i] if background_images else None
            result = self.apply_mask_ultra_fast(img, mask, original_size)
            
            if bg_img:
                result = self.process_with_background_ultra_fast(result, bg_img)
            
            return result
        
        process_args = [(i, input_images[i], masks_gpu[i], original_sizes[i]) for i in range(batch_size)]
        results = list(self._thread_pool.map(process_single_item, process_args))
        
        postprocess_time = time.perf_counter() - postprocess_start
        total_time = time.perf_counter() - start_time
        
        logger.info(f"FP16 Batch - Size: {batch_size}, "
                   f"Inference: {preprocess_inference_time:.3f}s, "
                   f"Postprocess: {postprocess_time:.3f}s, "
                   f"Total: {total_time:.3f}s, "
                   f"Per image: {total_time/batch_size:.3f}s")
        
        return results
    
    def remove_background(self, input_image: Image.Image, 
                         background_image: Image.Image = None) -> Image.Image:
        """
        Remove background from a single image.
        
        Args:
            input_image: PIL Image to process
            background_image: Optional background image to apply
            
        Returns:
            Processed PIL Image
        """
        start_time = time.perf_counter()
        
        original_size = input_image.size
        
        img_data = self.preprocess_ultra_fast(input_image, original_size)
        preprocess_time = time.perf_counter() - start_time
        
        inference_start = time.perf_counter()
        mask_gpu = self.predict_mask_gpu(img_data)
        inference_time = time.perf_counter() - inference_start
        
        postprocess_start = time.perf_counter()
        result_image = self.apply_mask_ultra_fast(input_image, mask_gpu, original_size)
        postprocess_time = time.perf_counter() - postprocess_start
        
        if background_image:
            compose_start = time.perf_counter()
            result_image = self.process_with_background_ultra_fast(result_image, background_image)
            compose_time = time.perf_counter() - compose_start
        else:
            compose_time = 0
            
        total_time = time.perf_counter() - start_time
        
        logger.info(f"FP16 Timing - Preprocess: {preprocess_time:.3f}s, "
                   f"Inference: {inference_time:.3f}s, "
                   f"Postprocess: {postprocess_time:.3f}s, "
                   f"Compose: {compose_time:.3f}s, "
                   f"Total: {total_time:.3f}s")
        
        return result_image
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'inputs') and hasattr(self, 'outputs') and hasattr(self, 'stream'):
                free_buffers(self.inputs, self.outputs, self.stream)
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True)
        except:
            pass