# Docker Setup for scFeatureLens

This directory provides Docker configuration for running scFeatureLens in a completely isolated, reproducible environment.

## 🐳 Quick Start with Docker

### Build and Run

```bash
# Build the Docker image
docker build -t scfeaturelens .

# Run basic example
docker run --rm scfeaturelens

# Run with custom data (mount your data directory)
docker run --rm -v /path/to/your/data:/app/data -v /path/to/results:/app/results scfeaturelens \
    python -m scFeatureLens.cli /app/data/embeddings.npy --output-dir /app/results
```

### Interactive Development

```bash
# Run interactive container for development
docker run -it --rm -v $(pwd):/app scfeaturelens bash

# Inside container, run any scFeatureLens commands
python -m scFeatureLens.example --example basic
python -m scFeatureLens.cli --help
```

## 🏗️ Building Options

### Standard Build

```bash
docker build -t scfeaturelens .
```

### GPU Support (if needed)

```dockerfile
# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# ... rest of Dockerfile
```

```bash
# Build with GPU support
docker build -f Dockerfile.gpu -t scfeaturelens:gpu .

# Run with GPU
docker run --rm --gpus all scfeaturelens:gpu
```

## 📦 Docker Compose Setup

Create `docker-compose.yml` for more complex setups:

```yaml
version: '3.8'

services:
  scfeaturelens:
    build: .
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./configs:/app/configs
    environment:
      - SC_MECHINTERP_DATA_DIR=/app/data
      - SC_MECHINTERP_CONFIG=/app/configs/analysis.yaml
    command: python -m scFeatureLens.cli /app/data/embeddings.npy --config /app/configs/analysis.yaml --output-dir /app/results

  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./results:/app/results
    command: jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app/notebooks
```

Run with:
```bash
docker-compose up
```

## 🔧 Customization

### Custom Configuration

```bash
# Mount config directory
docker run --rm \
    -v $(pwd)/configs:/app/configs \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    scfeaturelens \
    python -m scFeatureLens.cli /app/data/embeddings.npy \
    --config /app/configs/my_config.yaml \
    --output-dir /app/results
```

### Development Mode

```bash
# Mount source code for development
docker run -it --rm \
    -v $(pwd):/app \
    -w /app \
    scfeaturelens \
    bash

# Inside container, make changes and test immediately
```

## 📊 Production Deployment

### Multi-stage Build for Smaller Images

```dockerfile
# Build stage
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
RUN pip install --user -e .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "scFeatureLens.example", "--example", "basic"]
```

### Environment Variables

```bash
# Set environment variables for production
docker run --rm \
    -e SC_MECHINTERP_LOG_LEVEL=INFO \
    -e SC_MECHINTERP_DATA_DIR=/app/data \
    -e SC_MECHINTERP_CACHE_DIR=/app/.cache \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/results:/app/results \
    scfeaturelens
```

## 🐙 Container Registry

### Publish to Registry

```bash
# Tag for registry
docker tag scfeaturelens:latest your-registry.com/scfeaturelens:latest

# Push to registry
docker push your-registry.com/scfeaturelens:latest

# Pull and run from registry
docker pull your-registry.com/scfeaturelens:latest
docker run --rm your-registry.com/scfeaturelens:latest
```

## 🔒 Security Best Practices

The Docker image follows security best practices:
- Runs as non-root user
- Minimal base image
- No unnecessary packages
- Health checks included
- Proper file permissions

## 📋 Troubleshooting

### Common Issues

#### Permission Issues
```bash
# Fix file permissions
docker run --rm -v $(pwd):/app scfeaturelens chown -R $(id -u):$(id -g) /app/results
```

#### Memory Issues
```bash
# Increase Docker memory limit
docker run --rm -m 8g scfeaturelens
```

#### GPU Issues
```bash
# Check GPU support
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi
```

### Debugging

```bash
# Debug container
docker run -it --rm --entrypoint bash scfeaturelens

# Check logs
docker logs container_name

# Inspect container
docker inspect scfeaturelens
```

## 🚀 Performance Tips

- Use multi-stage builds for smaller images
- Mount data volumes instead of copying large files
- Use .dockerignore to exclude unnecessary files
- Consider using Docker BuildKit for faster builds
- Use specific base image versions for reproducibility

## 📝 Example Usage Scripts

### Batch Processing

```bash
#!/bin/bash
# batch_process.sh

DATA_DIR="/path/to/data"
RESULTS_DIR="/path/to/results"

for embedding_file in "$DATA_DIR"/*.npy; do
    filename=$(basename "$embedding_file" .npy)
    docker run --rm \
        -v "$DATA_DIR":/app/data \
        -v "$RESULTS_DIR":/app/results \
        scfeaturelens \
        python -m scFeatureLens.cli "/app/data/$filename.npy" \
        --output-dir "/app/results/$filename"
done
```

### Continuous Integration

```yaml
# .github/workflows/docker.yml
name: Docker Build and Test

on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t scfeaturelens .
      
      - name: Test Docker image
        run: docker run --rm scfeaturelens python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Test passed')"
      
      - name: Run example
        run: docker run --rm scfeaturelens
```
