from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# Load pre-trained ensemble model
try:
    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_loaded = True
    model_name = type(model).__name__
except:
    model_loaded = False
    model_name = "No Model"

# Load sample dataset for demonstration
data = load_breast_cancer()
feature_names = data.feature_names.tolist()

@app.route('/')
def index():
    return render_template('index.html', 
                         model_loaded=model_loaded,
                         model_name=model_name,
                         feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get features from form
        features = []
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            features.append(value)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        # Get prediction label
        prediction_label = 'Malignant' if prediction == 0 else 'Benign'
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_label': prediction_label,
            'probability_malignant': round(probability[0], 3),
            'probability_benign': round(probability[1], 3),
            'model_used': model_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/model_comparison')
def model_comparison():
    """Return sample model comparison data"""
    comparison_data = {
        'Random Forest': 0.964,
        'XGBoost': 0.971,
        'LightGBM': 0.968,
        'Voting Hard': 0.973,
        'Voting Soft': 0.975,
        'Stacking': 0.978
    }
    return jsonify(comparison_data)

if __name__ == '__main__':
    app.run(debug=True)