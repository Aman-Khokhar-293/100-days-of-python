from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# Sample training data: [area, bedrooms, bathrooms]
X = np.array([
    [1000, 2, 1],
    [1500, 3, 2],
    [2000, 4, 3],
    [2500, 4, 2],
    [3000, 5, 3]
])
y = np.array([100000, 150000, 200000, 230000, 280000])  # prices

model = LinearRegression()
model.fit(X, y)

with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as house_price_model.pkl")
