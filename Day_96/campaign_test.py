import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Generate synthetic data for two campaigns
np.random.seed(42)
campaign_A = np.random.normal(52, 10, 100)   # Conversion rates campaign A
campaign_B = np.random.normal(56, 9, 100)    # Conversion rates campaign B

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(campaign_A, campaign_B)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.4f}")

# Visualization
sns.boxplot(data=[campaign_A, campaign_B])
plt.xticks([0, 1], ['Campaign A', 'Campaign B'])
plt.title('Conversion Rate Comparison')
plt.ylabel('Conversion Rate')
plt.show()

if p_value < 0.05:
    print("✅ Significant difference between the two campaigns.")
else:
    print("❌ No significant difference between the two campaigns.")