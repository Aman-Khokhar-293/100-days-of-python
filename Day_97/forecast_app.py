from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    steps = int(request.form['months'])
    forecast = model.forecast(steps=steps)
    return render_template('index.html', prediction=forecast.to_list())

if __name__ == "__main__":
    app.run(debug=True)
