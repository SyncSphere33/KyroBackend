FROM python:3.10

# Install system dependencies required by OpenCV and ML/video libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["gunicorn", "app:app"]
