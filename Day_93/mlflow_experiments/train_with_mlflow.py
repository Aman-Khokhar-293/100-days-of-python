import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_sample_data():
    """Create sample classification dataset"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_redundant=2,
        n_informative=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, X, y, feature_names

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    return metrics, y_pred

def plot_feature_importance(model, feature_names, model_name):
    """Plot and save feature importance"""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_imp['feature'], feature_imp['importance'])
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        
        # Save plot
        plot_path = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    return None

def run_experiment():
    """Run ML experiment with multiple models and track with MLflow"""
    
    # Set experiment name
    mlflow.set_experiment("Classification_Model_Comparison")
    
    # Create sample data
    df, X, y, feature_names = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and track each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset_size", len(X_train))
            mlflow.log_param("num_features", X_train.shape[1])
            
            # Train model
            if model_name == 'SVM':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
                metrics, _ = evaluate_model(model, X_test_scaled, y_test)
            else:
                model.fit(X_train, y_train)
                metrics, predictions = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name.lower().replace(' ', '_')}_model")
            
            # Log feature importance plot
            if model_name != 'SVM':
                plot_path = plot_feature_importance(model, feature_names, model_name)
                if plot_path:
                    mlflow.log_artifact(plot_path)
            
            # Log sample predictions
            sample_preds = pd.DataFrame({
                'actual': y_test[:10],
                'predicted': predictions[:10]
            })
            sample_preds.to_csv(f"sample_predictions_{model_name.replace(' ', '_').lower()}.csv", index=False)
            mlflow.log_artifact(f"sample_predictions_{model_name.replace(' ', '_').lower()}.csv")
            
            print(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    print("🚀 Starting MLflow Experiment Tracking Demo")
    print("===========================================")
    run_experiment()
    print("🎉 All experiments completed! Run 'mlflow ui' to view results.")