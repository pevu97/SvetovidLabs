# Svetovid — reproducible CPU environment for training and inference
FROM python:3.12-slim

WORKDIR /app

# System deps required by opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch keeps the image small; remaining deps from PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY config.py train.py run_inference.py simulate_transmission.py generate_report.py ./
COPY svetovid/ ./svetovid/
COPY best_autoencoder.pth .

# Mount your image directory to /app/data at runtime:
#   docker run --rm -v $(pwd)/data:/app/data svetovid
CMD ["python", "run_inference.py"]
