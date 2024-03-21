import ray
from ray import serve
import requests, json
from starlette.requests import Request
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Response, Form
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Image Processing Service")

@serve.deployment(ray_actor_options={'num_gpus': 1})
@serve.ingress(app)
class Chat:
    def __init__(self, model: str):
        # configure stateful elements of our service such as loading a model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model =  AutoModelForSeq2SeqLM.from_pretrained(model).to(0)

    @app.get("/")
    async def checkhealth(self):
        return Response(status_code=200)
    
    @app.post("/")
    async def get_response(self, user_input: str) -> str:
        # this method receives calls directly (from Python) or from __call__ (from HTTP)
        #history.append(user_input)
        # the history is client-side state and will be a list of raw strings;
        # for the default config of the model and tokenizer, history should be joined with '</s><s>'
        inputs = self._tokenizer(user_input, return_tensors='pt').to(0)
        reply_ids = self._model.generate(**inputs, max_new_tokens=500)
        response = self._tokenizer.batch_decode(reply_ids.cpu(), skip_special_tokens=True)[0]
        return Response(response)
    
    
chat = Chat.bind(model='facebook/blenderbot-400M-distill')

