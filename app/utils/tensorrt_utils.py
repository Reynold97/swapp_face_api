import os
import numpy as np
import tensorrt as trt
import ctypes
from tensorrt_bindings import Logger
from cuda.bindings import driver as cuda, runtime as cudart
from typing import Optional, List, Union


def check_cuda_err(err):
    """Check CUDA error and raise RuntimeError if not successful."""
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    """Wrapper for CUDA calls with error checking."""
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


class HostDeviceMem:
    """Wrapper for host and device memory allocation."""
    
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        
        # Handle FP16 ctypes conversion
        if dtype == np.dtype(np.float16):
            pointer_type = ctypes.POINTER(ctypes.c_uint16)
            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            self._host = self._host.view(np.float16)
        else:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            if self.host.dtype == np.float16 and data.dtype != np.float16:
                data = data.astype(np.float16)
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        """Free CUDA memory."""
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    """Allocate buffers for TensorRT engine inputs and outputs."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(cudart.cudaStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    
    for binding in tensor_names:
        if profile_idx is not None:
            shape = engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        else:
            shape = engine.get_tensor_shape(binding)
            
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, but no profile was specified.")
        
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        if trt_type == trt.DataType.HALF:
            dtype = np.dtype(np.float16)
            bindingMemory = HostDeviceMem(size, dtype)
        elif trt.nptype(trt_type):
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else:
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        bindings.append(int(bindingMemory.device))

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    
    return inputs, outputs, bindings, stream


def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t):
    """Free all allocated buffers and destroy stream."""
    for mem in inputs + outputs:
        mem.free()
    cuda_call(cudart.cudaStreamDestroy(stream))


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    """Copy data from host to device."""
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    """Copy data from device to host."""
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    """Base inference function with async execution."""
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    execute_async_func()
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    cuda_call(cudart.cudaStreamSynchronize(stream))
    return [out.host for out in outputs]


def do_inference(context, engine, bindings, inputs, outputs, stream, batch_size=1):
    """Execute inference with TensorRT context."""
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)

    num_io = engine.num_io_tensors
    for i in range(num_io):
        tensor_name = engine.get_tensor_name(i)
        context.set_tensor_address(tensor_name, bindings[i])
        
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(tensor_name, (batch_size, 3, 1024, 1024))
            
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


def load_engine(engine_path: str):
    """Load serialized TensorRT engine from file."""
    if not os.path.exists(engine_path):
        raise ValueError(f"Engine file does not exist: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return engine_data


def convert_onnx_to_engine_fp16(onnx_filename: str, engine_filename: str = "engine_fp16.trt", max_batch_size: int = 20):
    """Convert ONNX model to TensorRT FP16 engine."""
    logger = Logger(Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)
    
    # Force FP16 completely
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    logger.log(trt.Logger.INFO, "Pure FP16 precision enforced")
    
    # Optimized memory for FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    
    # Force all layers to FP16
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer.precision = trt.DataType.HALF
        layer.set_output_type(0, trt.DataType.HALF)
    
    logger.log(trt.Logger.INFO, "Parsing ONNX for pure FP16 compilation")
    with open(onnx_filename, 'rb') as model:
        if not parser.parse(model.read()):
            logger.log(trt.Logger.ERROR, "Failed to parse onnx file")
            for err in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(err))
            raise RuntimeError("parse onnx file error")
    
    # Set input tensor to FP16
    input_tensor = network.get_input(0)
    input_tensor.dtype = trt.DataType.HALF
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input_image", 
                     (1, 3, 1024, 1024),
                     (max_batch_size//2, 3, 1024, 1024),
                     (max_batch_size, 3, 1024, 1024))
    
    config.add_optimization_profile(profile)
    
    logger.log(trt.Logger.INFO, f"Building pure FP16 engine with batch 1-{max_batch_size}")
    engine_bytes = builder.build_serialized_network(network, config)
    
    with open(engine_filename, 'wb') as f:
        f.write(engine_bytes)
    logger.log(trt.Logger.INFO, f"Pure FP16 engine saved to {engine_filename}")
    
    return engine_bytes


def sigmoid(x):
    """Numerically stable sigmoid function."""
    pos_mask = x >= 0
    neg_mask = x < 0
    result = np.zeros_like(x)
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))
    return result