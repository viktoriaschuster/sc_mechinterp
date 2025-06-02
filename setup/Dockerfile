# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 scuser && chown -R scuser:scuser /app
USER scuser

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY --chown=scuser:scuser . .

# Install the package in development mode
RUN pip install --no-cache-dir --user -e .

# Create data and results directories
RUN mkdir -p data results example_data

# Set environment variables
ENV PATH="/home/scuser/.local/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV SC_MECHINTERP_DATA_DIR="/app/data"
ENV SC_MECHINTERP_CACHE_DIR="/app/.cache"

# Verify installation
RUN python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ scFeatureLens imported successfully')"

# Default command
CMD ["python", "-m", "scFeatureLens.example", "--example", "basic"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from tools.scFeatureLens import SCFeatureLensPipeline" || exit 1

# Labels for metadata
LABEL maintainer="scFeatureLens Team"
LABEL description="Mechanistic interpretability tool for single-cell RNA-seq data"
LABEL version="1.0.0"
