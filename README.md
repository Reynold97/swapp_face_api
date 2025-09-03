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

## Server Deployment

### Server Requirements

To deploy this API on a cloud server, you'll need:

- **GPU Server**: A server instance with NVIDIA GPU support
- **NVIDIA Drivers**: CUDA 11.8 or compatible drivers installed
- **Docker**: Docker engine with NVIDIA Container Toolkit for GPU support
- **Network Access**: Firewall configured to allow traffic on port 8000

### Deployment Steps

#### 1. Server Setup
- Create a GPU-enabled server instance from your cloud provider
- Choose an instance with NVIDIA drivers and CUDA 11.8 pre-installed (recommended)
- Ensure the server has sufficient resources (minimum 8GB RAM, 16GB+ recommended)
- Configure the server with appropriate disk space for Docker images and models

#### 2. Install Docker and GPU Support
- Install Docker engine on your server
- Install NVIDIA Container Toolkit to enable GPU access in Docker containers
- Verify GPU access with `nvidia-smi` command
- Test Docker GPU integration

#### 3. Transfer Credentials
- Copy your `bucket_credentials.json` file to the server
- Create a credentials directory (e.g., `/home/username/credentials/`)
- Set appropriate file permissions (600) for security
- Verify the file is accessible and readable

#### 4. Configure Network Access
- Configure your server's firewall to allow incoming traffic on port 8000
- For cloud providers, create firewall rules to allow HTTP traffic on port 8000
- Note your server's external IP address for API access

#### 5. Deploy the Container
- Pull the Docker image from Docker Hub
- Run the container with GPU support and credentials mounted
- Verify the container starts successfully and models load on GPU
- Test API access using the server's external IP

#### 6. Verification
- Access the Swagger documentation at `http://YOUR_SERVER_IP:8000/docs`
- Test the `/swap_img` endpoint with sample images
- Monitor container logs for any errors or warnings
- Verify GPU usage during face swapping operations

### Production Considerations

- **Security**: Restrict firewall rules to specific IP ranges when possible
- **Monitoring**: Set up logging and monitoring for the container
- **Backup**: Regularly backup your credentials and any persistent data
- **Updates**: Plan for periodic updates of the Docker image
- **Scaling**: Consider load balancing for multiple instances if needed

## Quick Start with Docker

### Option 1: Pull from Docker Hub (Recommended)

```bash
# Pull the pre-built image
docker pull docker pull reynoldoramas/swapp_face_blackbox:latest

# Run with GPU support (mount your GCP credentials)
docker run --gpus all -p 8000:8000 \
  -v /path/to/your/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  [YOUR_DOCKERHUB_USERNAME]/swapp-face-api:latest

# Run without GPU (CPU only)
docker run -d -p 8000:8000 \
  --gpus all \
  -v /home/username/credentials/bucket_credentials.json:/app/credentials/bucket_credentials.json:ro \
  --name swapp-face-api \
  reynoldoramas/swapp_face_blackbox:latest
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

**4. Cannot access API externally**
```
Connection refused or timeout when accessing http://SERVER_IP:8000
```
- Check firewall rules allow traffic on port 8000
- Verify the container is running and listening on 0.0.0.0:8000
- Ensure the server's external IP is correctly configured

**5. Container exits immediately**
```
Container starts but stops immediately
```
- Check container logs with `docker logs [container_name]`
- Verify credentials file is properly mounted and accessible
- Ensure sufficient disk space for models and temporary files
