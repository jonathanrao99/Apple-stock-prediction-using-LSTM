#!/bin/bash

# Apple Stock Predictor Deployment Script
# This script helps deploy the application using Docker

set -e

echo "🚀 Starting Apple Stock Predictor Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data models results logs

# Build and run with Docker Compose
print_status "Building and starting the application..."
docker-compose up --build -d

# Wait for the application to start
print_status "Waiting for the application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    print_status "✅ Application is running successfully!"
    echo ""
    echo "🌐 Access the application at: http://localhost:8501"
    echo "📊 The web interface is now available in your browser"
    echo ""
    echo "📋 Useful commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop application: docker-compose down"
    echo "  - Restart application: docker-compose restart"
    echo ""
else
    print_error "❌ Application failed to start. Check the logs with: docker-compose logs"
    exit 1
fi

print_status "Deployment completed successfully! 🎉"
