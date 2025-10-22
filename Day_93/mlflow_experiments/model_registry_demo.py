import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

def model_registry_demo():
    """Demonstrate MLflow Model Registry functionality"""
    
    mlflow.set_experiment("Model_Registry_Demo")
    
    # Create dataset
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train multiple versions of the model
    model_versions = [
        {"n_estimators": 50, "max_depth": 5, "version": "v1"},
        {"n_estimators": 100, "max_depth": 7, "version": "v2"},
        {"n_estimators": 200, "max_depth": None, "version": "v3"}
    ]
    
    registered_models = []
    
    for params in model_versions:
        with mlflow.start_run(run_name=f"Model_{params['version']}"):
            # Train model
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "random_forest_model",
                registered_model_name="Classification_Model"
            )
            
            registered_models.append({
                'version': params['version'],
                'accuracy': accuracy,
                'f1_score': f1,
                'run_id': mlflow.active_run().info.run_id
            })
            
            print(f"✅ Registered {params['version']} - Accuracy: {accuracy:.4f}")
    
    # Demonstrate model staging
    print("\n🎭 Model Staging Demo:")
    
    # Transition best model to Production
    best_model = max(registered_models, key=lambda x: x['f1_score'])
    
    # Note: In practice, you'd use MLflow Client API for transitions
    print(f"🏆 Best model: {best_model['version']} (F1: {best_model['f1_score']:.4f})")
    print("📝 In production, you would transition this model to 'Production' stage")
    
    return registered_models

if __name__ == "__main__":
    model_registry_demo()