FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
# 1. Install base PyTorch first (matching your target system, e.g., CPU or CUDA)
# For CPU-only docker containers:
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR if your Docker base image supports CUDA 11.8:
# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Next, install the pre-compiled PyG binaries that match your Torch version
RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# (Change "+cpu.html" to "+cu118.html" if you are using CUDA)

# 3. Finally, install the rest of your project dependencies
RUN pip install --no-cache-dir -r requirements.txt
 

# Install additional dependencies for API and dashboard
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    streamlit==1.31.0 \
    plotly==5.18.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data artifacts data/graphs

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden)
CMD ["python", "gnn_pipeline.py"]