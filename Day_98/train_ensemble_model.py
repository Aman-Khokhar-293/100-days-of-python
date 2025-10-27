import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    StackingClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedEnsemble:
    def __init__(self):
        self.models = {}
        self.ensemble_models = {}
        
    def generate_sample_data(self):
        """Generate synthetic classification dataset"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=1,
            random_state=42
        )
        return X, y
    
    def load_real_data(self):
        """Load real dataset for demonstration"""
        data = load_breast_cancer()
        return data.data, data.target
    
    def create_base_models(self):
        """Create diverse base models for ensemble"""
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'LightGBM': LGBMClassifier(random_state=42),
            'CatBoost': CatBoostClassifier(silent=True, random_state=42)
        }
    
    def create_ensemble_models(self, base_models):
        """Create ensemble models using stacking and voting"""
        # Voting Classifier - Hard Voting
        voting_hard = VotingClassifier(
            estimators=[
                ('rf', base_models['Random Forest']),
                ('xgb', base_models['XGBoost']),
                ('lgbm', base_models['LightGBM'])
            ],
            voting='hard'
        )
        
        # Voting Classifier - Soft Voting
        voting_soft = VotingClassifier(
            estimators=[
                ('rf', base_models['Random Forest']),
                ('xgb', base_models['XGBoost']),
                ('lgbm', base_models['LightGBM'])
            ],
            voting='soft'
        )
        
        # Stacking Classifier
        stacking = StackingClassifier(
            estimators=[
                ('rf', base_models['Random Forest']),
                ('xgb', base_models['XGBoost']),
                ('lgbm', base_models['LightGBM'])
            ],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        self.ensemble_models = {
            'Voting Hard': voting_hard,
            'Voting Soft': voting_soft,
            'Stacking': stacking
        }
    
    def evaluate_models(self, X_train, X_test, y_train, y_test):
        """Evaluate all models and return performance metrics"""
        results = {}
        
        # Evaluate base models
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
        
        # Evaluate ensemble models
        for name, model in self.ensemble_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            print(f"{name}: {accuracy:.4f}")
        
        return results
    
    def plot_comparison(self, results):
        """Plot model performance comparison and save to file"""
        plt.figure(figsize=(12, 6))
        models = list(results.keys())
        accuracies = list(results.values())
        
        colors = ['skyblue' if 'Voting' not in model and 'Stacking' not in model 
                 else 'lightcoral' for model in models]
        
        bars = plt.bar(models, accuracies, color=colors, alpha=0.7)
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        # Save instead of showing to avoid blocking/hanging in non-GUI envs
        plt.savefig('comparison.png', dpi=200, bbox_inches='tight')
        plt.close()
    
    def save_best_model(self, results):
        """Save the best performing model"""
        best_model_name = max(results, key=results.get)
        
        if best_model_name in self.models:
            best_model = self.models[best_model_name]
        else:
            best_model = self.ensemble_models[best_model_name]
        
        with open('ensemble_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print(f"✅ Best model saved: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")
        return best_model_name

def main():
    print("🚀 Day 98: Advanced Ensemble Methods")
    print("=" * 50)
    
    # Initialize ensemble system
    ensemble_system = AdvancedEnsemble()
    
    # Load data
    print("📊 Loading dataset...")
    X, y = ensemble_system.load_real_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create models
    print("\n🤖 Creating base models...")
    ensemble_system.create_base_models()
    
    print("\n🎯 Creating ensemble models...")
    ensemble_system.create_ensemble_models(ensemble_system.models)
    
    # Evaluate all models
    print("\n📈 Evaluating models...")
    results = ensemble_system.evaluate_models(X_train, X_test, y_train, y_test)
    
    # Plot comparison
    print("\n📊 Plotting results...")
    ensemble_system.plot_comparison(results)
    
    # Save best model
    print("\n💾 Saving best model...")
    best_model_name = ensemble_system.save_best_model(results)
    
    print(f"\n✅ Ensemble training completed!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best accuracy: {results[best_model_name]:.4f}")

if __name__ == "__main__":
    main()