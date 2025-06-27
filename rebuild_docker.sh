#!/bin/bash

# Rebuild Docker image with updated dependencies
# This script rebuilds the churn prediction API with the latest scikit-learn version

echo "🔄 Rebuilding Docker image with updated scikit-learn version..."

# Stop any running containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Remove old image to force rebuild
echo "🗑️  Removing old image..."
docker rmi churn-api:latest 2>/dev/null || true

# Build new image
echo "🏗️  Building new image..."
docker-compose build --no-cache

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for health check
echo "⏳ Waiting for health check..."
sleep 10

# Test health endpoint
echo "🏥 Testing health endpoint..."
curl -s http://localhost:8000/health | jq .

echo "✅ Docker rebuild complete!"
echo "📊 API available at: http://localhost:8000"
echo "📚 Documentation at: http://localhost:8000/docs"