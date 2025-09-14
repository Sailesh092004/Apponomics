#!/bin/bash

# Apponomics Deployment Script

set -e

echo "🚀 Deploying Apponomics..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models config

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting Apponomics..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Apponomics is running successfully!"
    echo "🌐 Access the application at: http://localhost:8501"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi

echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
