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
COPY deployment/models/churn_model.pkl ./models/churn_model.pkl

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

# Set entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]