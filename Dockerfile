FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch CPU-only first (avoids pulling the 2 GB GPU build + triton).
# torch-geometric is installed after so it can find torch at build time.
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torch-geometric>=2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY *.py /app/

RUN mkdir -p /app/src/data/graphs \
    && mkdir -p /app/src/data/graphs_hetero \
    && mkdir -p /app/src/artifacts \
    && mkdir -p /app/src/artifacts_hetero \
    && mkdir -p /app/artifacts

ENV PYTHONPATH=/app:/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 8501

CMD ["python", "src/model_api.py"]
