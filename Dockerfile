FROM python:3.10

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "app:app"]
