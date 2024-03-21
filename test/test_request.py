import requests, json
from starlette.requests import Request


sample_json = '{ "user_input" : "hello", "history" : [] }'
response = requests.post("http://localhost:22060/", json = sample_json).json()
print(response) 