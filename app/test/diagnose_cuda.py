import ray
import onnxruntime as ort
import os
import subprocess

@ray.remote(num_gpus=1)
def diagnose_gpu():
    def run_command(command):
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
            return result.stdout.strip()
        except Exception as e:
            return f"Error running command: {e}"

    info = {
        "ONNX Runtime version": ort.__version__,
        "Available providers": ort.get_available_providers(),
        "CUDA available": 'CUDAExecutionProvider' in ort.get_available_providers(),
        "Environment variables": {
            "LD_LIBRARY_PATH": os.environ.get('LD_LIBRARY_PATH', 'Not set'),
            "CUDA_HOME": os.environ.get('CUDA_HOME', 'Not set')
        },
        "CUDA version": run_command("nvcc --version"),
        "GPU information": run_command("nvidia-smi"),
        "CUDA libraries": run_command("ldconfig -p | grep libcuda"),
    }
    return info

# Initialize Ray (if not already initialized)
if not ray.is_initialized():
    ray.init()

# Run the diagnosis on a worker with GPU
result = ray.get(diagnose_gpu.remote())

# Print the results
for key, value in result.items():
    print(f"{key}:")
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            print(f"  {sub_key}: {sub_value}")
    else:
        print(f"  {value}")

# Shut down Ray
ray.shutdown()