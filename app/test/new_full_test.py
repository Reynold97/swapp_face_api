#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime
import os
from pathlib import Path

# ==================== CONFIGURATION ====================
# Leave empty to skip tests that require multiple people
SINGLE_PERSON_IMAGE_1 = "1.jpg"  # Image with one person
SINGLE_PERSON_IMAGE_2 = "2.png"  # Image with one person
MULTI_PERSON_IMAGE_1 = ""  # Image with multiple people
MULTI_PERSON_IMAGE_2 = ""  # Image with multiple people

# Background removal test image (local file path)
BG_REMOVAL_TEST_IMAGE = "app/test/test_data/frame.png"

# API Configuration
API_BASE_URL = "http://localhost:8000"
OUTPUT_DIR = "app/test/test_results"  # Directory to save results
ENDPOINT_TYPE = "url"  # "url" or "img" - use "url" for GCP filenames, "img" for local files

# ==================== END CONFIGURATION ====================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_header(message):
    """Print a header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")

def log_test(test_name):
    """Print a test name"""
    print(f"{Colors.CYAN}{Colors.BOLD}[TEST] {test_name}{Colors.END}")

def log_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def log_error(message):
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def log_info(message):
    """Print an info message"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.END}")

def log_warning(message):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_info(f"Output directory: {OUTPUT_DIR}")

def save_image(response, filename):
    """Save image response to file"""
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "wb") as f:
        f.write(response.content)
    log_success(f"Saved: {filename}")
    return output_path

def test_face_swap_url(face_filename, model_filename, mode="one_to_one", 
                       direction="left_to_right", use_codeformer=False,
                       codeformer_fidelity=0.5, test_name=""):
    """Test face swap using URL endpoint (GCP filenames)"""
    url = f"{API_BASE_URL}/swap_url"
    
    payload = {
        "face_filename": face_filename,
        "model_filenames": [model_filename],
        "options": {
            "mode": mode,
            "direction": direction,
            "use_codeformer": use_codeformer,
            "codeformer_fidelity": codeformer_fidelity,
            "background_enhance": True,
            "face_upsample": True,
            "upscale": 2
        }
    }
    
    log_test(test_name or f"Face Swap - {mode}")
    log_info(f"Source Face: {face_filename}")
    log_info(f"Target Model: {model_filename}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            log_success(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            
            # Display URLs only (don't save)
            if "data" in result:
                log_info(f"Results: {len(result['data'])} image(s)")
                for i, img_data in enumerate(result["data"], 1):
                    if "url" in img_data:
                        log_success(f"  → URL {i}: {img_data['url']}")
                    elif "filename" in img_data:
                        log_success(f"  → Filename {i}: {img_data['filename']}")
                    else:
                        log_success(f"  → Result {i}: Image data received")
            else:
                log_warning("No 'data' field in response")
            
            return True
        else:
            log_error(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            log_error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"Exception after {elapsed:.2f}s: {str(e)}")
        return False

def test_face_swap_img(face_path, model_path, mode="one_to_one",
                       direction="left_to_right", use_codeformer=False,
                       codeformer_fidelity=0.5, test_name=""):
    """Test face swap using IMG endpoint (local files)"""
    url = f"{API_BASE_URL}/swap_img"
    
    params = {
        "mode": mode,
        "direction": direction,
        "use_codeformer": use_codeformer,
        "codeformer_fidelity": codeformer_fidelity,
        "background_enhance": True,
        "face_upsample": True,
        "upscale": 2
    }
    
    log_test(test_name or f"Face Swap - {mode}")
    log_info(f"Source Face: {face_path}")
    log_info(f"Target Model: {model_path}")
    
    start_time = time.time()
    try:
        with open(face_path, "rb") as face_file, open(model_path, "rb") as model_file:
            files = {
                "face": face_file,
                "model": model_file
            }
            
            response = requests.post(url, params=params, files=files, timeout=120)
            
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            log_success(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            
            # Save image to test results folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            enhancer_suffix = "_cf" if use_codeformer else ""
            filename = f"swap_{mode}{enhancer_suffix}_{timestamp}.png"
            save_image(response, filename)
            
            return True
        else:
            log_error(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            log_error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"Exception after {elapsed:.2f}s: {str(e)}")
        return False

def test_background_removal(image_path, background_path=None, test_name=""):
    """Test background removal using the /api/remove-bg endpoint"""
    url = f"{API_BASE_URL}/api/remove-bg"
    
    log_test(test_name or "Background Removal")
    log_info(f"Image: {image_path}" + (f", Background: {background_path}" if background_path else ""))
    
    start_time = time.time()
    try:
        files = {"image": open(image_path, "rb")}
        
        if background_path:
            files["background"] = open(background_path, "rb")
        
        response = requests.post(url, files=files, timeout=120)
        
        # Close files
        for f in files.values():
            f.close()
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            log_success(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            
            # Save result image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bg_suffix = "_with_bg" if background_path else "_no_bg"
            filename = f"bg_removal{bg_suffix}_{timestamp}.png"
            save_image(response, filename)
            
            return True
        else:
            log_error(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            log_error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"Exception after {elapsed:.2f}s: {str(e)}")
        return False

def test_batch_background_removal(image_paths, background_paths=None, test_name=""):
    """Test batch background removal using the /api/batch-remove-bg endpoint"""
    url = f"{API_BASE_URL}/api/batch-remove-bg"
    
    log_test(test_name or "Batch Background Removal")
    log_info(f"Images: {len(image_paths)} files")
    
    start_time = time.time()
    try:
        files = [("images", open(img_path, "rb")) for img_path in image_paths]
        
        if background_paths:
            bg_files = [("backgrounds", open(bg_path, "rb")) for bg_path in background_paths]
            files.extend(bg_files)
        
        response = requests.post(url, files=files, timeout=120)
        
        # Close files
        for _, f in files:
            f.close()
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            log_success(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            log_info(f"  Processed: {result.get('processed_count', 0)}/{result.get('total_count', 0)}")
            
            return True
        else:
            log_error(f"Status: {response.status_code} - Time: {elapsed:.2f}s")
            log_error(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        elapsed = time.time() - start_time
        log_error(f"Exception after {elapsed:.2f}s: {str(e)}")
        return False

def run_all_tests():
    """Run all configured tests"""
    ensure_output_dir()
    
    log_header("FACE SWAP & BACKGROUND REMOVAL API TEST SUITE")
    log_info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"API Base URL: {API_BASE_URL}")
    log_info(f"Endpoint Type: {ENDPOINT_TYPE}")
    
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0
    }
    
    # Determine which test function to use for face swap
    if ENDPOINT_TYPE == "url":
        swap_test_func = test_face_swap_url
    else:
        swap_test_func = test_face_swap_img
    
    # ========== BASIC TESTS (Always run) ==========
    log_header("BASIC TESTS")
    
    if SINGLE_PERSON_IMAGE_1 and SINGLE_PERSON_IMAGE_2:
        # Test 1: Basic one-to-one swap
        results["total"] += 1
        if swap_test_func(SINGLE_PERSON_IMAGE_1, SINGLE_PERSON_IMAGE_2, 
                         mode="one_to_one", test_name="Basic One-to-One Swap"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
    
    # ========== BACKGROUND REMOVAL TESTS ==========
    if BG_REMOVAL_TEST_IMAGE and os.path.exists(BG_REMOVAL_TEST_IMAGE):
        log_header("BACKGROUND REMOVAL TESTS")
        
        # Test 2: Background removal without background
        results["total"] += 1
        if test_background_removal(BG_REMOVAL_TEST_IMAGE, test_name="Background Removal (No Background)"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
        
        # Test 3: Background removal with background
        if ENDPOINT_TYPE == "img" and SINGLE_PERSON_IMAGE_2 and os.path.exists(SINGLE_PERSON_IMAGE_2):
            results["total"] += 1
            if test_background_removal(BG_REMOVAL_TEST_IMAGE, SINGLE_PERSON_IMAGE_2,
                           test_name="Background Removal (With Background)"):
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            print()  # spacing
        
        # Test 4: Batch background removal (if we have the test image)
        if os.path.exists(BG_REMOVAL_TEST_IMAGE):
            # Use the same image twice for batch test
            results["total"] += 1
            if test_batch_background_removal([BG_REMOVAL_TEST_IMAGE, BG_REMOVAL_TEST_IMAGE],
                                            test_name="Batch Background Removal (2 images)"):
                results["passed"] += 1
            else:
                results["failed"] += 1
            
            print()  # spacing
    
    # ========== ADVANCED TESTS (Only if multi-person images provided) ==========
    if MULTI_PERSON_IMAGE_1 and MULTI_PERSON_IMAGE_2:
        log_header("ADVANCED TESTS (Multi-Person)")
        
        # Test 5: One-to-many swap
        results["total"] += 1
        if swap_test_func(SINGLE_PERSON_IMAGE_1, MULTI_PERSON_IMAGE_1,
                         mode="one_to_many", test_name="One-to-Many Swap"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
        
        # Test 6: Sorted swap (left to right)
        results["total"] += 1
        if swap_test_func(MULTI_PERSON_IMAGE_1, MULTI_PERSON_IMAGE_2,
                         mode="sorted", direction="left_to_right", 
                         test_name="Sorted Swap (Left to Right)"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
        
        # Test 7: Sorted swap (right to left)
        results["total"] += 1
        if swap_test_func(MULTI_PERSON_IMAGE_1, MULTI_PERSON_IMAGE_2,
                         mode="sorted", direction="right_to_left",
                         test_name="Sorted Swap (Right to Left)"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
        
        # Test 8: Similarity swap
        results["total"] += 1
        if swap_test_func(MULTI_PERSON_IMAGE_1, MULTI_PERSON_IMAGE_2,
                         mode="similarity", test_name="Similarity Swap"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
    
    # ========== ENHANCEMENT TESTS ==========
    if SINGLE_PERSON_IMAGE_1 and SINGLE_PERSON_IMAGE_2:
        log_header("ENHANCEMENT TESTS (CodeFormer)")
        
        # Test 9: Basic swap with CodeFormer
        results["total"] += 1
        if swap_test_func(SINGLE_PERSON_IMAGE_1, SINGLE_PERSON_IMAGE_2,
                         mode="one_to_one", use_codeformer=True,
                         codeformer_fidelity=0.5, test_name="One-to-One with CodeFormer"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
    
    if MULTI_PERSON_IMAGE_1 and MULTI_PERSON_IMAGE_2:
        # Test 10: Sorted swap with CodeFormer
        results["total"] += 1
        if swap_test_func(MULTI_PERSON_IMAGE_1, MULTI_PERSON_IMAGE_2,
                         mode="sorted", direction="left_to_right",
                         use_codeformer=True, codeformer_fidelity=0.7,
                         test_name="Sorted Swap with CodeFormer"):
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        print()  # spacing
    
    # ========== RESULTS SUMMARY ==========
    log_header("TEST RESULTS SUMMARY")
    
    log_info(f"Total Tests: {results['total']}")
    log_success(f"Passed: {results['passed']}")
    if results['failed'] > 0:
        log_error(f"Failed: {results['failed']}")
    else:
        log_info(f"Failed: {results['failed']}")
    
    success_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0
    
    if success_rate == 100:
        log_success(f"\n🎉 All tests passed! Success rate: {success_rate:.1f}%")
    elif success_rate >= 80:
        log_warning(f"\n⚠️  Most tests passed. Success rate: {success_rate:.1f}%")
    else:
        log_error(f"\n❌ Many tests failed. Success rate: {success_rate:.1f}%")
    
    log_info(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Results saved in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        log_warning("\n\nTests interrupted by user")
    except Exception as e:
        log_error(f"\n\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()