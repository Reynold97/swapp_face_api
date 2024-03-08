import ray
from ray import serve

ray.init()  # Initialize Ray
serve.start()  # Start Ray Serve