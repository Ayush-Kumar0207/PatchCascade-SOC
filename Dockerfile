# =============================================================================
# PatchCascade SOC - Dockerfile
# =============================================================================
# OpenEnv-compliant container for Hugging Face Spaces deployment.
# Runs the FastAPI environment server on port 8000.
#
# Build:  docker build -t patchcascade-soc .
# Run:    docker run -p 8000:8000 patchcascade-soc
# =============================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY environment.py .
COPY server.py .
COPY client.py .
COPY inference.py .

# Copy metadata
COPY openenv.yaml .

# Expose the server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
