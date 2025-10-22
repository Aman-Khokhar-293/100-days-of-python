import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def hyperparameter_tuning_demo():
    """Demonstrate hyperparameter tuning with MLflow tracking"""
    
    mlflow.set_experiment("Hyperparameter_Tuning")
    
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_redundant=3,
        n_informative=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    print("🔍 Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    # Log all parameter combinations
    for i, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(run_name=f"RF_HP_{i}", nested=True):
            # Log parameters
            for param, value in params.items():
                mlflow.log_param(param, value)
            
            # Log cross-validation scores
            mean_score = grid_search.cv_results_['mean_test_score'][i]
            std_score = grid_search.cv_results_['std_test_score'][i]
            
            mlflow.log_metric("cv_mean_f1", mean_score)
            mlflow.log_metric("cv_std_f1", std_score)
            
            # Train model with these parameters and log test performance
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_f1", test_f1)
            
            # Log model if it's the best so far
            if i == np.argmax(grid_search.cv_results_['mean_test_score']):
                mlflow.sklearn.log_model(model, "best_tuned_model")
                mlflow.set_tag("best_model", "true")
    
    print(f"🎯 Best parameters: {grid_search.best_params_}")
    print(f"🏆 Best CV score: {grid_search.best_score_:.4f}")

if __name__ == "__main__":
    hyperparameter_tuning_demo()