#!/usr/bin/env python3
"""
Demo script for training and testing the packaged ML model
"""

from model_packaging.model import MLModel
from model_packaging.utils import create_sample_data, plot_feature_importance

def main():
    print("🚀 ML Model Packaging Demo")
    print("=" * 40)
    
    # Create sample data
    print("📊 Creating sample data...")
    df, X, y = create_sample_data()
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    # Initialize and train model
    print("\n🎯 Training ML model...")
    model = MLModel()
    results = model.train(X, y)
    
    print(f"✅ Training completed!")
    print(f"📈 Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"📊 Test Accuracy: {results['test_accuracy']:.4f}")
    
    # Plot feature importance
    print("\n📊 Plotting feature importance...")
    plot_feature_importance(feature_names, results['feature_importance'])
    
    # Save the model
    print("\n💾 Saving model...")
    model.save_model('trained_model.pkl')
    print("✅ Model saved as 'trained_model.pkl'")
    
    print("\n🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()