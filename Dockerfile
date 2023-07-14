FROM python:3.11
WORKDIR /ML-Flask
COPY . .
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --bind 0.0.0.0:$PORT main:app