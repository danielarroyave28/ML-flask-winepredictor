FROM python:3.11
COPY ../ML-Flask
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn main:app --bind 0.0.0.0:$PORT main:app