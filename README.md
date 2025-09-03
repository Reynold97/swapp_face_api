# Face Swap API

A high-performance face swapping API built with FastAPI that supports multiple swap modes and enhancement options using deep learning models.

## Features

- **Multiple Swap Modes:**
  - `one_to_one`: Standard single face swap (default)
  - `one_to_many`: Apply one source face to all target faces
  - `sorted`: Apply faces in spatial order (left-to-right or right-to-left)
  - `similarity`: Match source faces to target faces by similarity

- **Enhancement Options:**
  - Standard face enhancement with GFPGAN
  - CodeFormer enhancement with configurable fidelity
  - Background enhancement and face upsampling
  - Configurable upscale factors

- **Deployment Ready:**
  - Dockerized with NVIDIA CUDA support
  - FastAPI with automatic API documentation
  - Health checks and monitoring ready

## Prerequisites

### Google Cloud Platform Setup

The API uses Google Cloud Storage for the `/swap_url` endpoint. **Note**: The `/swap_img` endpoint works without GCP setup as it uses direct file uploads.

For `/swap_url` endpoint, you need to set up:

1. **Create a GCP Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or use an existing one

2. **Create a Storage Bucket**
   - Navigate to Cloud Storage
   - Create a bucket named `anyscale_tmp_faces`
   - Choose your preferred region

3. **Create a Service Account**
   - Go to IAM & Admin → Service Accounts
   - Click "Create Service Account"
   - Name it (e.g., "face-swap-api")
   - Grant the following roles:
     - `Storage Object Admin` (for read/write access to the bucket)
     - `Storage Legacy Bucket Reader` (for listing bucket contents)

4. **Generate Credentials**
   - Click on the created service account
   - Go to the "Keys" tab
   - Click "Add Key" → "Create new key" → "JSON"
   - Download the JSON file and rename it to `bucket_credentials.json`

## Quick Start with Docker

### Option 1: Pull from Docker Hub (Recommended)

```bash
# Pull the pre-built image
docker pull [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest

# Run with GPU support (mount your GCP credentials)
docker run --gpus all -p 8000:8000 \
  -v /path/to/your/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest

# Run without GPU (CPU only)
docker run -p 8000:8000 \
  -v /path/to/your/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone <repository-url>
cd swapp_face_api

# Build the Docker image
docker build -t swapp-face-api .

# Run with GPU support (mount your GCP credentials)
docker run --gpus all -p 8000:8000 \
  -v /path/to/your/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  swapp-face-api

# Run without GPU (CPU only)
docker run -p 8000:8000 \
  -v /path/to/your/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  swapp-face-api
```

**Important:** Replace `/path/to/your/bucket_credentials.json` with the actual path to your downloaded GCP credentials file.

### Quick Start without GCP (Only /swap_img endpoint)

If you only want to use the `/swap_img` endpoint (direct file uploads), you can run without mounting credentials:

```bash
# Pull and run without GCP credentials
docker pull [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest
docker run --gpus all -p 8000:8000 [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest
```

**Note:** The `/swap_url` endpoint will not work without proper GCP credentials.

## API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## API Endpoints

### POST /swap_img
Upload images directly for face swapping.

**Parameters:**
- `face`: Source face image file
- `model`: Target image file
- `mode`: Swap mode (one_to_one, one_to_many, sorted, similarity)
- `direction`: Direction for sorted mode (left_to_right, right_to_left)
- `use_codeformer`: Use CodeFormer for enhancement (boolean)
- `codeformer_fidelity`: Balance between quality and fidelity (0-1)
- `background_enhance`: Enhance background (boolean)
- `face_upsample`: Upsample faces (boolean)
- `upscale`: Upscale factor (integer)

### POST /swap_url
Swap faces using images from GCP storage.

**Body:**
```json
{
  "face_filename": "source_face.jpg",
  "model_filenames": ["target1.jpg", "target2.jpg"],
  "options": {
    "mode": "one_to_one",
    "direction": "left_to_right",
    "use_codeformer": false,
    "codeformer_fidelity": 0.5,
    "background_enhance": true,
    "face_upsample": true,
    "upscale": 2
  }
}
```

## Example Usage

### cURL Example
```bash
curl -X POST "http://localhost:8000/swap_img" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "face=@source_face.jpg" \
  -F "model=@target_image.jpg" \
  -F "mode=one_to_one" \
  -F "use_codeformer=true" \
  -F "codeformer_fidelity=0.7"
```

### Python Example
```python
import requests

url = "http://localhost:8000/swap_img"
files = {
    'face': open('source_face.jpg', 'rb'),
    'model': open('target_image.jpg', 'rb')
}
data = {
    'mode': 'one_to_one',
    'use_codeformer': True,
    'codeformer_fidelity': 0.7
}

response = requests.post(url, files=files, data=data)
with open('result.png', 'wb') as f:
    f.write(response.content)
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8 support (recommended)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: ~5GB for models and dependencies

## Local Development

### Installation
```bash
pip install -r requirements.txt
```

```bash
pip install -r codeformer_requirements.txt
```

### Run locally
```bash
uvicorn app.api.main:app --reload --port 8000
```

## Performance Notes

Typical processing times on GPU:
- Face Analysis: ~350ms
- Face Swapping: ~1.7s
- Enhancement: ~1.5s
- **Total**: ~4.8s per swap

## Health Check

The container includes a health check endpoint:
```bash
curl http://localhost:8000/docs
```

## Troubleshooting

### Common Issues

**1. GCP Credentials Error**
```
Error: Could not load credentials from /app/credentials/bucket_credentials.json
```
- Ensure the credentials file is properly mounted using `-v`
- Check that the file path exists on your host system
- Verify the JSON file is valid and not corrupted

**2. Bucket Access Error**
```
Error: 403 Forbidden when accessing bucket
```
- Verify your service account has `Storage Object Admin` role
- Ensure the bucket name matches exactly (`tmp_faces`)
- Check that the bucket exists in your GCP project

**3. GPU not detected**
```
CUDA device not found, running on CPU
```
- Install NVIDIA Docker toolkit: `distribution=$(. /etc/os-release;echo $ID$VERSION_ID)`
- Restart Docker after installation
- Use `--gpus all` flag when running the container

## License

[Add your license information here]