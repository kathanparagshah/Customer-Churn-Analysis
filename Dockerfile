# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and constraints files for better caching
COPY requirements.txt constraints.txt ./

# Install Python dependencies with pip cache mount and constraints
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p logs models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]