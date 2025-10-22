#!/bin/bash

echo "🚀 Starting MLflow Tracking Server"
echo "==================================="

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

echo "MLflow UI available at: http://localhost:5000"