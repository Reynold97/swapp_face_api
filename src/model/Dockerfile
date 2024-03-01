FROM python:3.10

WORKDIR /roop

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg

RUN pip install --verbose -r requirements.txt

RUN python run.py --target ./.github/examples/model.png  --source ./.github/examples/face.jpeg -o ./.github/examples/swapped.png --execution-provider cpu --frame-processor face_swapper face_enhancer