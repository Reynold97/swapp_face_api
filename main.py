import sys
import uvicorn

from src.controllers import swap_face

from fastapi import FastAPI, status
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import ray
from ray import serve

app = FastAPI()

allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add controllers
app.include_router(swap_face.router)

@app.get("/")
async def root():
    return Response(status_code=200)

@serve.deployment
@serve.ingress(app)
class FastAPIWrapper:
    pass

if __name__ == '__main__':
    if sys.argv[1] == 'server':
        uvicorn.run(app=app, host='0.0.0.0', port=5001, log_level="info")
