# Face Swap API Documentation

This documentation provides comprehensive information about the Face Swap API, which allows you to swap faces in images with various options and modes.

## API Endpoints

The API provides two main endpoints for face swapping:

1. `/swap_url` - For swapping faces using GCP-stored images
2. `/swap_img` - For swapping faces using directly uploaded images

## 1. Swap URL Endpoint

### Endpoint: `/swap_url`

This endpoint allows you to swap faces between images stored in your GCP bucket.

#### Request Format

**HTTP Method**: POST

**Content-Type**: application/json

**Request Body**:

```json
{
  "face_filename": "source-face.jpg",
  "model_filenames": ["target-image1.jpg", "target-image2.jpg"],
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

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| face_filename | string | Yes | The filename of the source face image in the GCP bucket |
| model_filenames | array of strings | Yes | List of target image filenames in the GCP bucket |
| options | object | No | Configuration options for the face swap operation |

### Options Object

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| mode | string | "one_to_one" | Face swap mode. Options: "one_to_one", "one_to_many", "sorted", "similarity" |
| direction | string | "left_to_right" | Direction for sorted mode. Options: "left_to_right", "right_to_left" |
| use_codeformer | boolean | false | Whether to use CodeFormer for enhancement |
| codeformer_fidelity | number (0.0-1.0) | 0.5 | Balance between quality and fidelity (0 for better quality, 1 for better identity) |
| background_enhance | boolean | true | Whether to enhance the background (CodeFormer only) |
| face_upsample | boolean | true | Whether to upsample the faces |
| upscale | integer (1-4) | 2 | The upscale factor for enhancement |

#### Response Format

For successful operations with all images:

```json
{
  "urls": ["processed-image1.jpg", "processed-image2.jpg"]
}
```

For partially successful operations (some images failed):

```json
{
  "urls": ["processed-image1.jpg"],
  "failed": [
    {
      "filename": "target-image2.jpg",
      "error": "No face detected in the target image"
    }
  ]
}
```

#### Status Codes

- **200 OK**: All images processed successfully
- **206 Partial Content**: Some images processed, some failed
- **400 Bad Request**: Invalid request or no faces detected
- **500 Internal Server Error**: Server-side error

## 2. Swap Image Endpoint

### Endpoint: `/swap_img`

This endpoint allows you to swap faces by directly uploading the source and target images.

#### Request Format

**HTTP Method**: POST

**Content-Type**: multipart/form-data

**Request Body**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| face | file | Yes | The source face image file |
| model | file | Yes | The target image file |
| mode | string | No | Face swap mode (one_to_one, one_to_many, sorted, similarity) |
| direction | string | No | Direction for sorted mode (left_to_right, right_to_left) |
| use_codeformer | boolean | No | Whether to use CodeFormer for enhancement |
| codeformer_fidelity | number | No | Balance between quality and fidelity (0-1) |
| background_enhance | boolean | No | Whether to enhance the background |
| face_upsample | boolean | No | Whether to upsample the faces |
| upscale | integer | No | The upscale factor for enhancement (1-4) |

#### Response Format

The response is the processed image in PNG format.

#### Status Codes

- **200 OK**: Image processed successfully
- **400 Bad Request**: Invalid request or no faces detected
- **500 Internal Server Error**: Server-side error

## Swap Modes Explained

The API supports four different modes for face swapping:

### 1. one_to_one (Default)

This mode performs a standard face swap between one source face and one target face. If multiple faces are present in the images, only the first detected face in each image is used.

**Example use case**: Swap a single person's face onto another person.

### 2. one_to_many

This mode takes the first face detected in the source image and applies it to all faces detected in the target image.

**Example use case**: Put the same face on multiple people in a group photo.

### 3. sorted

This mode sorts faces by their spatial position in the images and maps them accordingly. This allows control over which face goes where in multi-face scenarios:

- **left_to_right**: Source faces (sorted left-to-right) are mapped to target faces (sorted left-to-right or right-to-left based on the direction parameter)
- **right_to_left**: Source faces (sorted left-to-right) are mapped to target faces (sorted right-to-left)

**Example use case**: Swap faces between two couples, matching their positions.

### 4. similarity

This mode matches source faces to target faces based on their feature similarity. The system attempts to find the best matches based on facial features.

**Example use case**: Swap faces between groups of people while preserving most appropriate pairings.

## Enhancement Options

### CodeFormer Enhancement

Setting `use_codeformer` to `true` activates advanced face enhancement:

- `codeformer_fidelity` controls balance between quality and identity preservation
  - Values closer to 0 produce higher quality but may alter identity more
  - Values closer to 1 preserve more identity but with potentially less enhancement

### Background and Upscaling

- `background_enhance`: When true, enhances the non-face portions of the image
- `face_upsample`: When true, applies additional upsampling to faces
- `upscale`: Controls the overall upscaling factor (1-4)

## Error Handling

The API returns structured error responses when problems occur:

```json
{
  "error": "Bad Request",
  "message": "No face detected in the provided source image"
}
```

Common error scenarios:

1. No face detected in source image
2. No face detected in target image
3. Invalid parameters
4. Server processing errors

## Example Requests

### Example 1: Basic one-to-one swap

```bash
curl -X POST http://api.example.com/swap_url \
  -H "Content-Type: application/json" \
  -d '{
    "face_filename": "person1.jpg",
    "model_filenames": ["person2.jpg"],
    "options": {
      "mode": "one_to_one"
    }
  }'
```

### Example 2: One-to-many with CodeFormer enhancement

```bash
curl -X POST http://api.example.com/swap_url \
  -H "Content-Type: application/json" \
  -d '{
    "face_filename": "person1.jpg",
    "model_filenames": ["group_photo.jpg"],
    "options": {
      "mode": "one_to_many",
      "use_codeformer": true,
      "codeformer_fidelity": 0.7,
      "upscale": 2
    }
  }'
```

### Example 3: Sorted mode (left-to-right)

```bash
curl -X POST http://api.example.com/swap_url \
  -H "Content-Type: application/json" \
  -d '{
    "face_filename": "couple1.jpg",
    "model_filenames": ["couple2.jpg"],
    "options": {
      "mode": "sorted",
      "direction": "left_to_right"
    }
  }'
```

### Example 4: Direct file upload

```bash
curl -X POST http://api.example.com/swap_img \
  -F "face=@person1.jpg" \
  -F "model=@person2.jpg" \
  -F "mode=one_to_one" \
  -F "use_codeformer=true" \
  -F "codeformer_fidelity=0.6"
```

## Known Limitations

1. The API requires clear, front-facing faces for best results
2. Very small faces in images may not be detected
3. Extreme lighting conditions or occlusions may affect face detection and swapping quality
4. The maximum recommended image resolution is 4K