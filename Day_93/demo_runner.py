#!/usr/bin/env python3
"""
Main script to run all MLflow demos
"""

import subprocess
import sys
import time

def run_demos():
    print("🚀 MLflow Demo Runner - Day 93")
    print("===============================")
    
    # Run basic experiment tracking
    print("\n1. Running Basic Experiment Tracking...")
    from mlflow_experiments.train_with_mlflow import run_experiment
    run_experiment()
    
    # Run hyperparameter tuning
    print("\n2. Running Hyperparameter Tuning Demo...")
    from mlflow_experiments.hyperparameter_tuning import hyperparameter_tuning_demo
    hyperparameter_tuning_demo()
    
    # Run model registry demo
    print("\n3. Running Model Registry Demo...")
    from mlflow_experiments.model_registry_demo import model_registry_demo
    model_registry_demo()
    
    print("\n🎉 All demos completed!")
    print("\n📊 To view results, run: mlflow ui")
    print("🌐 Then open: http://localhost:5000")

if __name__ == "__main__":
    run_demos()