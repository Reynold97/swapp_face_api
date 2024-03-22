import onnxruntime as ort
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
    
print(ort.get_all_providers())

import tensorflow as tf
print("cuDNN Version:", tf.config.experimental.list_physical_devices('GPU'))