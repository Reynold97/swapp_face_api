import requests

url = "http://localhost:8000/swap_url_codeformer"

# Query parameters
params = {
    "face_filename": "!1.jpg",
    "fidelity_weight": 0.5,
    "background_enhance": True,
    "face_upsample": True
}

# The body should be just a list of model filenames
model_filenames = ["!2.png"]

response = requests.post(url, params=params, json=model_filenames)
print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")