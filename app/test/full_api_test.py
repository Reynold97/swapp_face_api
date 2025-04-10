import requests
import argparse
import json

def test_swap_url(face_filename, model_filename, 
               mode="one_to_one", 
               direction="left_to_right",
               use_codeformer=False,
               codeformer_fidelity=0.5, 
               background_enhance=True, 
               face_upsample=True, 
               upscale=2):
    """
    Test the swap_url endpoint using GCP-stored images.
    
    Args:
        face_filename: Source face image filename in GCP
        model_filename: Target image filename in GCP
        mode: Swap mode (one_to_one, one_to_many, sorted, similarity)
        direction: Direction for sorted mode (left_to_right, right_to_left)
        use_codeformer: Whether to use CodeFormer for enhancement
        codeformer_fidelity: Balance between quality and fidelity (0-1)
        background_enhance: Whether to enhance the background
        face_upsample: Whether to upsample the faces
        upscale: Upscale factor for enhancement
    """
    url = "http://localhost:8000/swap_url"
    
    # Prepare the request payload
    payload = {
        "face_filename": face_filename,
        "model_filenames": [model_filename],
        "options": {
            "mode": mode,
            "direction": direction,
            "use_codeformer": use_codeformer,
            "codeformer_fidelity": codeformer_fidelity,
            "background_enhance": background_enhance,
            "face_upsample": face_upsample,
            "upscale": upscale
        }
    }
    
    print(f"Testing swap_url with the following parameters:")
    print(json.dumps(payload, indent=2))
    
    response = requests.post(url, json=payload)
    
    print(f"\nResponse Status Code: {response.status_code}")
    try:
        print(f"Response Content: {json.dumps(response.json(), indent=2)}")
    except json.JSONDecodeError:
        print(f"Response Text: {response.text}")
    
    return response

def test_swap_img(face_image_path, model_image_path,
               mode="one_to_one", 
               direction="left_to_right",
               use_codeformer=False,
               codeformer_fidelity=0.5, 
               background_enhance=True, 
               face_upsample=True, 
               upscale=2):
    """
    Test the swap_img endpoint with file uploads.
    
    Args:
        face_image_path: Path to the source face image file
        model_image_path: Path to the target image file
        mode: Swap mode (one_to_one, one_to_many, sorted, similarity)
        direction: Direction for sorted mode (left_to_right, right_to_left)
        use_codeformer: Whether to use CodeFormer for enhancement
        codeformer_fidelity: Balance between quality and fidelity (0-1)
        background_enhance: Whether to enhance the background
        face_upsample: Whether to upsample the faces
        upscale: Upscale factor for enhancement
    """
    url = "http://localhost:8000/swap_img"
    
    # Prepare the query parameters
    params = {
        "mode": mode,
        "direction": direction,
        "use_codeformer": use_codeformer,
        "codeformer_fidelity": codeformer_fidelity,
        "background_enhance": background_enhance,
        "face_upsample": face_upsample,
        "upscale": upscale
    }
    
    # Prepare the file uploads
    files = {
        "face": open(face_image_path, "rb"),
        "model": open(model_image_path, "rb")
    }
    
    print(f"Testing swap_img with the following parameters:")
    print(f"  Query params: {json.dumps(params, indent=2)}")
    print(f"  Source face: {face_image_path}")
    print(f"  Target model: {model_image_path}")
    
    response = requests.post(url, params=params, files=files)
    
    print(f"\nResponse Status Code: {response.status_code}")
    
    # If successful, save the image to a file
    if response.status_code == 200:
        output_path = "result.png"
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Saved result to {output_path}")
    else:
        try:
            print(f"Response Content: {json.dumps(response.json(), indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
    
    # Close file handles
    for f in files.values():
        f.close()
    
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the face swap API endpoints")
    parser.add_argument("--endpoint", choices=["url", "img"], required=True, 
                        help="Which endpoint to test (url or img)")
    
    # Common parameters
    parser.add_argument("--mode", choices=["one_to_one", "one_to_many", "sorted", "similarity"], 
                        default="one_to_one", help="Swap mode")
    parser.add_argument("--direction", choices=["left_to_right", "right_to_left"], 
                        default="left_to_right", help="Direction for sorted mode")
    parser.add_argument("--use_codeformer", action="store_true", help="Use CodeFormer for enhancement")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, 
                        help="Balance between quality and fidelity (0-1)")
    parser.add_argument("--background_enhance", action="store_true", default=True, 
                        help="Enhance background (CodeFormer only)")
    parser.add_argument("--no_background_enhance", action="store_false", dest="background_enhance", 
                        help="Don't enhance background")
    parser.add_argument("--face_upsample", action="store_true", default=True, 
                        help="Upsample faces (CodeFormer only)")
    parser.add_argument("--no_face_upsample", action="store_false", dest="face_upsample", 
                        help="Don't upsample faces")
    parser.add_argument("--upscale", type=int, default=2, choices=[1, 2, 3, 4], 
                        help="Upscale factor for enhancement")
    
    # URL endpoint specific args
    parser.add_argument("--face_filename", type=str, help="Source face filename in GCP")
    parser.add_argument("--model_filename", type=str, help="Target image filename in GCP")
    
    # IMG endpoint specific args
    parser.add_argument("--face_path", type=str, help="Path to source face image file")
    parser.add_argument("--model_path", type=str, help="Path to target image file")
    
    args = parser.parse_args()
    
    # Common options for both endpoints
    common_opts = {
        "mode": args.mode,
        "direction": args.direction,
        "use_codeformer": args.use_codeformer,
        "codeformer_fidelity": args.codeformer_fidelity,
        "background_enhance": args.background_enhance,
        "face_upsample": args.face_upsample,
        "upscale": args.upscale
    }
    
    if args.endpoint == "url":
        if not args.face_filename or not args.model_filename:
            parser.error("--face_filename and --model_filename are required for URL endpoint")
        
        test_swap_url(
            face_filename=args.face_filename,
            model_filename=args.model_filename,
            **common_opts
        )
    elif args.endpoint == "img":
        if not args.face_path or not args.model_path:
            parser.error("--face_path and --model_path are required for IMG endpoint")
        
        test_swap_img(
            face_image_path=args.face_path,
            model_image_path=args.model_path,
            **common_opts
        )