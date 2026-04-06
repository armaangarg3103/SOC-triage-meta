# Use a specific Python 3.11 slim image
FROM python:3.11-slim

# System deps for httpx and uvicorn
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency spec first (layer cache)
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy the rest of the source
COPY . .

# Environment defaults (non-secret)
ENV ENVIRONMENT=production
ENV MAX_EPISODES=1000
ENV ENABLE_WEB_INTERFACE=true

# Expose port 7860 (matches openenv.yaml and uvicorn command)
EXPOSE 7860

# Healthcheck on the same port as the runtime
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server on 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
