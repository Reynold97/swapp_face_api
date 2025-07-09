import requests

url = "http://localhost:8000/swap_url"

payload = {
        "model_filenames": ["2.png"],  # Replace with your actual model filenames
        "face_filename": "1.jpg"  # Replace with your actual face filename
}

response = requests.post(url, json=payload)
print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")