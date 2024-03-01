FROM python:3.10

WORKDIR /roop-api

RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD [ "python", "src/main.py", "server" ] 