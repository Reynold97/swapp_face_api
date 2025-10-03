# Face Swap API

A high-performance face swapping API built with Ray Serve and FastAPI, featuring multiple swap modes and advanced face enhancement capabilities.

## Overview

This API provides sophisticated face swapping functionality with support for:
- **Multiple swap modes**: One-to-one, one-to-many, sorted, and similarity-based matching
- **Advanced enhancement**: GFPGAN and CodeFormer integration for high-quality results
- **Scalable architecture**: Built on Ray Serve for horizontal scaling
- **GPU acceleration**: Optimized for CUDA-enabled environments
- **Cloud storage**: Integrated with Google Cloud Storage for image management

## Features

### Swap Modes
- **One-to-One**: Standard face swap between single faces
- **One-to-Many**: Apply one source face to all faces in target image
- **Sorted**: Spatial ordering (left-to-right or right-to-left) face mapping
- **Similarity**: AI-driven matching based on facial feature similarity

### Enhancement Options
- **GFPGAN**: Traditional face enhancement
- **CodeFormer**: Advanced enhancement with quality/fidelity balance
- **Background enhancement**: Optional background upscaling
- **Face refinement**: Iterative swapping for improved results

### Core Components
- **APIIngress**: FastAPI endpoints for HTTP requests
- **SwapProcessor**: Orchestrates different swap modes and workflows  
- **FaceAnalyzer**: Face detection and recognition using RetinaFace + ArcFace
- **FaceSwapper**: Face swapping using InSwapper model
- **FaceEnhancer**: Face enhancement using GFPGAN
- **CodeFormerEnhancer**: Advanced enhancement using CodeFormer
- **GCPImageManager**: Google Cloud Storage integration

## Deployment Guide

### Prerequisites

1. **Anyscale Account**: Create an account and set up an Anyscale Cloud
   - Follow the official documentation: https://docs.anyscale.com/
   - Create a cloud integration with your GCP project

### GCP Setup

Either use the GCP cli or console, refer to GCP documentation.

#### 1. Create Storage Bucket
```bash
# Create the required bucket
gsutil mb gs://anyscale_tmp_faces

# Set bucket permissions for public read (optional, for testing)
gsutil iam ch allUsers:objectViewer gs://anyscale_tmp_faces
```

#### 2. Create Service Account
```bash
# Create service account
gcloud iam service-accounts create anyscale-face-swap \
    --description="Service account for Face Swap API" \
    --display-name="Anyscale Face Swap"

# Grant storage permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:anyscale-face-swap@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Create and download key
gcloud iam service-accounts keys create anyscale_bucket_credentials.json \
    --iam-account=anyscale-face-swap@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

#### 3. Configure Credentials
```bash
# Move the credentials file to the configs folder
mv anyscale_bucket_credentials.json app/configs/
```

### Anyscale Deployment

#### 1. Create Workspace
1. In the Anyscale Console, create a new Workspace
2. Use the base container image: `reynoldoramas/anyscale-swapface:py3.10-devcu11.8-cudnn8-ray2.46.0`
3. Ensure your workspace has access to your GCP-integrated cloud

#### 2. Clone and Deploy
```bash
# Clone the repository in your Anyscale workspace
git clone <your-repo-url>
cd <repo-name>

# Deploy the service
anyscale service deploy -f app/configs/service_config_gpu_load.yaml -n gpu_server

# Check service status
anyscale service status --name=gpu_server
```

#### 3. Service Management
```bash
# View service logs
anyscale service logs --name=gpu_server --follow

# Update service
anyscale service deploy -f app/configs/service_config_gpu_load.yaml -n gpu_server --in-place

# Terminate service
anyscale service terminate --name=gpu_server
```

## API Usage

### Endpoints

#### 1. URL-based Face Swap
**POST** `/swap_url`

Swap faces using images stored in your GCP bucket.

```bash
curl -X POST "https://your-service-url/swap_url" \
  -H "Content-Type: application/json" \
  -d '{
    "face_filename": "source_face.jpg",
    "model_filenames": ["target_image.jpg"],
    "options": {
      "mode": "one_to_one",
      "use_codeformer": true,
      "codeformer_fidelity": 0.5,
      "upscale": 2
    }
  }'
```

#### 2. Direct Image Upload
**POST** `/swap_img`

Upload images directly for face swapping.

```bash
curl -X POST "https://your-service-url/swap_img" \
  -F "face=@source_face.jpg" \
  -F "model=@target_image.jpg" \
  -F "mode=one_to_one" \
  -F "use_codeformer=true" \
  -F "codeformer_fidelity=0.6"
```

### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | string | Swap mode: `one_to_one`, `one_to_many`, `sorted`, `similarity` |
| `direction` | string | For sorted mode: `left_to_right`, `right_to_left` |
| `use_codeformer` | boolean | Use CodeFormer for enhancement |
| `codeformer_fidelity` | float (0-1) | Balance quality vs identity preservation |
| `background_enhance` | boolean | Enhance background (CodeFormer only) |
| `face_upsample` | boolean | Upsample faces |
| `upscale` | integer (1-4) | Upscaling factor |

### Response Format

**Success (200)**:
```json
{
  "urls": ["processed_image_1.jpg", "processed_image_2.jpg"]
}
```

**Partial Success (206)**:
```json
{
  "urls": ["processed_image_1.jpg"],
  "failed": [
    {
      "filename": "failed_image.jpg", 
      "error": "No face detected"
    }
  ]
}
```

## Performance Testing

Use the included Locust configuration for load testing:

```bash
# Simple load test
locust -f app/test/locust_face_swap_simple.py \
  --host=https://your-service-url/ \
  -u 5 -r 1 -t 30s
```

## Configuration

The service supports multiple deployment configurations:

- `service_config_gpu.yaml`: Standard GPU deployment
- `service_config_gpu_load.yaml`: High-load GPU deployment  
- `service_config_cpu.yaml`: CPU-only deployment (for testing)

Adjust resource allocation, autoscaling parameters, and replica counts based on your needs.

## Monitoring

- **Service Health**: Built-in health checks for all components
- **Metrics**: Ray Dashboard provides detailed metrics
- **Logs**: Structured JSON logging for debugging

## Limitations

- Maximum recommended image resolution: 4K
- Requires clear, front-facing faces for best results
- Small or heavily occluded faces may not be detected
- Extreme lighting conditions may affect quality

## Documentation

For detailed information about Anyscale concepts:
- **Workspaces**: https://docs.anyscale.com/workspaces/
- **Services**: https://docs.anyscale.com/services/
- **Clouds**: https://docs.anyscale.com/clouds/
- **General Documentation**: https://docs.anyscale.com/

## Support

For technical issues:
1. Check service logs: `anyscale service logs --name=gpu_server`
2. Verify GCP credentials and permissions
3. Ensure proper image formats and face visibility
4. Review Anyscale resource allocation

## License

[Add your license information here]