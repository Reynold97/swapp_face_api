FROM python:3.10

WORKDIR /roop-api

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg

RUN pip install -r src/model/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5001

CMD [ "python", "src/main.py", "server" ]