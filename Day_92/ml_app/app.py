from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
from model import MLModel

app = Flask(__name__)

# Initialize model
model = MLModel()

@app.route('/')
def home():
    return jsonify({
        "message": "ML Dockerized App - Day 92",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "train": "/train (POST)",
            "info": "/info"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_trained": model.is_trained})

@app.route('/info')
def info():
    return jsonify({
        "app_name": "Dockerized ML Application",
        "version": "1.0.0",
        "author": "Aman Khokhar",
        "day": 92,
        "topic": "Docker for ML"
    })

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        
        if not data or 'X' not in data or 'y' not in data:
            return jsonify({"error": "Please provide 'X' and 'y' in request body"}), 400
        
        X = np.array(data['X'])
        y = np.array(data['y'])
        
        results = model.train(X, y)
        
        # Save model
        model.save_model('trained_model.joblib')
        
        return jsonify({
            "message": "Model trained successfully",
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model.is_trained:
            return jsonify({"error": "Model not trained yet. Please train first."}), 400
        
        data = request.get_json()
        
        if not data or 'X' not in data:
            return jsonify({"error": "Please provide 'X' in request body"}), 400
        
        X = np.array(data['X'])
        predictions = model.predict(X)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "count": len(predictions)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        if os.path.exists('trained_model.joblib'):
            model.load_model('trained_model.joblib')
            return jsonify({"message": "Model loaded successfully"})
        else:
            return jsonify({"error": "No saved model found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', False))