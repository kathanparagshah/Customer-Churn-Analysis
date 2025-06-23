#!/bin/bash

# Customer Churn Analysis - Quick Setup Script
# This script sets up the development environment for both frontend and backend

set -e  # Exit on any error

echo "🚀 Setting up Customer Churn Analysis Development Environment"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "frontend" ] || [ ! -d "deployment" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

if ! command_exists npm; then
    echo "❌ npm is required but not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup Python environment
echo "🐍 Setting up Python environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Python environment setup complete"

# Setup Frontend
echo "⚛️  Setting up Frontend environment..."

cd frontend

echo "Installing Node.js dependencies..."
npm install

echo "Setting up environment variables..."
if [ ! -f ".env.local" ]; then
    cp .env.example .env.local
    echo "📝 Created .env.local from .env.example"
    echo "   You can edit this file to customize your API URL"
fi

cd ..

echo "✅ Frontend environment setup complete"

# Setup Docker (optional)
echo "🐳 Checking Docker setup..."

if command_exists docker; then
    echo "✅ Docker is available"
    echo "   You can run 'docker-compose up --build' to start the full stack"
else
    echo "⚠️  Docker is not installed (optional)"
    echo "   Install Docker to use containerized deployment"
fi

# Create logs directory
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p models

echo ""
echo "🎉 Setup complete! Here's how to get started:"
echo "================================================"
echo ""
echo "🔧 Development Commands:"
echo "  Backend (API):     source venv/bin/activate && uvicorn deployment.app:app --reload"
echo "  Frontend (React):  cd frontend && npm run dev"
echo "  Full Stack:        docker-compose up --build"
echo ""
echo "🌐 URLs:"
echo "  Frontend:          http://localhost:3000"
echo "  Backend API:       http://localhost:8000"
echo "  API Documentation: http://localhost:8000/docs"
echo ""
echo "📚 Next Steps:"
echo "  1. Train a model:   python train_model.py"
echo "  2. Start backend:   source venv/bin/activate && uvicorn deployment.app:app --reload"
echo "  3. Start frontend:  cd frontend && npm run dev"
echo "  4. Open browser:    http://localhost:3000"
echo ""
echo "📖 For deployment instructions, see DEPLOYMENT.md"
echo "🐛 For troubleshooting, see README.md"
echo ""
echo "Happy coding! 🚀"