import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def train(self, X, y):
        """Train the model on given data"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.is_trained = True
        
        print(f"Training completed - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': self.model.feature_importances_.tolist(),
            'training_samples': len(X_train),
            'testing_samples': len(X_test)
        }
    
    def predict(self, X):
        """Make predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")