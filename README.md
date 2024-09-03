Installation 

pip install cupy-cuda11x==13.1.0 opencv-python-headless==4.9.0.80 onnx==1.16.0 onnxruntime-gpu==1.17.1 python-multipart
pip install cupy-cuda11x opencv-python-headless onnx onnxruntime python-multipart

serve run app/configs/serve_config_cpu.yaml

anyscale service deploy -f app/configs/service_config_cpu.yaml

anyscale service terminate --name=SwapFaceAPI-Service-Prod

locust -f app/test/locustfile.py

Example url call cpu

DOWNLOAD_IMAGE OK 210.7ms   4.4%
EXTRACT_FACES OK 354.5ms    7.4%
DOWNLOAD_IMAGE OK 63.7ms    1.34%
EXTRACT_FACES OK 352.3ms    7.4%
SWAP_FACE OK 1771.1ms       37%
EXTRACT_FACES OK 275.1ms    5.8%
ENHANCE_FACE OK 1518.4ms    32%
Total time: OK 4764.7ms     95%


curl -X POST "http://localhost:8000/swap_url2?face_filename=Reynold_Oramas.jpg" \
-H "Content-Type: application/json" \
-d '["samurai.png"]'

