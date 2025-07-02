# Multi-stage Docker build for Customer Churn Prediction API
# Stage 1: Builder stage
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt constraints.txt ./

# Install dependencies globally
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Stage 2: Production stage
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy global site-packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Create directories first
RUN mkdir -p logs reports/figures models

# Copy application files
COPY deployment/app.py .
COPY deployment/analytics_db.py .
COPY src/ ./src/
COPY app/ ./app/
COPY deployment/models/ ./models/

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/churn_model.pkl

# Expose port
EXPOSE 8000

# Health check - optimized for fast response
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Set entrypoint - use the new modular app structure directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]