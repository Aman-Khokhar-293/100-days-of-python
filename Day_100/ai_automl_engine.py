import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class AutoMLEngine:
    
    def __init__(self, task_type='auto'):
        self.task_type = task_type
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.best_score = 0
        self.model_results = {}
        
    def detect_task_type(self, y):
        # auto-detect classification vs regression based on unique values
        if self.task_type != 'auto':
            return self.task_type
            
        unique_values = len(np.unique(y))
        total_values = len(y)
        
        if unique_values < 20 or (unique_values / total_values) < 0.05:
            return 'classification'
        return 'regression'
    
    def preprocess_data(self, df, target_column):
        print("🔍 Preprocessing data...")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.task_type = self.detect_task_type(y)
        print(f"📊 Detected task type: {self.task_type.upper()}")
        
        # encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        X = X.fillna(X.mean())
        
        if self.task_type == 'classification' and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            self.label_encoders[target_column] = le
        
        return X, y
    
    def train_models(self, X, y):
        print("\n🤖 Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # train different models based on task type
        if self.task_type == 'classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
            }
            scoring_metric = accuracy_score
        else:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Linear Regression': LinearRegression()
            }
            scoring_metric = r2_score
        
        # train and compare all models
        for name, model in models.items():
            print(f"\n   Training {name}...")
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
            if self.task_type == 'classification':
                score = scoring_metric(y_test, predictions)
                self.model_results[name] = {
                    'model': model,
                    'accuracy': score,
                    'predictions': predictions
                }
                print(f"   ✓ {name} Accuracy: {score:.4f}")
            else:
                score = scoring_metric(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                self.model_results[name] = {
                    'model': model,
                    'r2_score': score,
                    'mse': mse,
                    'predictions': predictions
                }
                print(f"   ✓ {name} R² Score: {score:.4f}, MSE: {mse:.4f}")
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = name
        
        # get feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Best Model: {self.best_model_name} (Score: {self.best_score:.4f})")
        
        return X_test_scaled, y_test
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_model_summary(self):
        return {
            'task_type': self.task_type,
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'feature_importance': self.feature_importance,
            'all_results': self.model_results
        }


# demo function to test the AutoML engine
def demo_automl():
    from sklearn.datasets import load_iris, load_diabetes
    
    print("=" * 60)
    print("🚀 AI-POWERED AUTOML ENGINE DEMO")
    print("=" * 60)
    
    # test with iris dataset
    print("\n📌 CLASSIFICATION TASK (Iris Dataset)")
    print("-" * 60)
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    
    automl_clf = AutoMLEngine(task_type='auto')
    X, y = automl_clf.preprocess_data(df_iris, 'target')
    automl_clf.train_models(X, y)
    
    print("\n📊 Feature Importance:")
    print(automl_clf.feature_importance.head())
    
    # test with diabetes dataset
    print("\n\n📌 REGRESSION TASK (Diabetes Dataset)")
    print("-" * 60)
    diabetes = load_diabetes()
    df_diabetes = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df_diabetes['target'] = diabetes.target
    
    automl_reg = AutoMLEngine(task_type='auto')
    X, y = automl_reg.preprocess_data(df_diabetes, 'target')
    automl_reg.train_models(X, y)
    
    print("\n📊 Feature Importance:")
    print(automl_reg.feature_importance.head())


if __name__ == "__main__":
    demo_automl()
