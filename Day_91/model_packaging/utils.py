import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_sample_data(n_samples=1000, n_features=10):
    """Create sample classification data for demonstration"""
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=2,
        n_informative=8,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, X, y

def plot_feature_importance(feature_names, importance_scores):
    """Plot feature importance from trained model"""
    plt.figure(figsize=(10, 6))
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True)
    
    plt.barh(feature_imp_df['feature'], feature_imp_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()