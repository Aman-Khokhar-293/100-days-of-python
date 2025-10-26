import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load sample data
data = pd.read_csv('sales_data.csv')
sales = data['sales']

# Train ARIMA model
model = ARIMA(sales, order=(2,1,2))
model_fit = model.fit()

# Save model
pickle.dump(model_fit, open('model.pkl', 'wb'))
print("Model trained and saved successfully.")
