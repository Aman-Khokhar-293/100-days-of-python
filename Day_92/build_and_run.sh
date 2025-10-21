#!/bin/bash

echo "🚀 Building and Running Dockerized ML Application"
echo "=============================================="

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t ml-docker-app:day92 .

# Run the container
echo "🐳 Starting Docker container..."
docker run -d -p 5000:5000 --name ml-app-day92 ml-docker-app:day92

echo "✅ Application is running!"
echo "🌐 Access the application at: http://localhost:5000"
echo "📊 Health check: http://localhost:5000/health"
echo "ℹ️  App info: http://localhost:5000/info"

# Show running containers
echo ""
echo "📋 Running containers:"
docker ps | grep ml-app-day92

echo ""
echo "🛑 To stop the container: docker stop ml-app-day92"
echo "🗑️  To remove the container: docker rm ml-app-day92"