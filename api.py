from flask import Flask, Response, request
import pandas as pd
import os
from io import StringIO

from pandas.core.indexes.base import Index
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

training_data = pd.read_csv(os.path.join('data', 'auto-mpg.csv'))

trained_model = pd.read_pickle(os.path.join('models', 'baummethoden.pickle'))

prediction_data = pd.read_csv(os.path.join('data', 'prediction_input_mpg.csv'))

@app.route('/')
def main():
    return "<p>Hello, World!</p>"

@app.route('/training_data')
def get_training_data():
    return Response(training_data.to_json(), mimetype = 'application/json')

@app.route('/prediction_data')
def get_prediction_data():
    return Response(prediction_data.to_json(), mimetype = 'application/json')

@app.route('/predict')
def predict():
    zylinder = request.args.get('zylinder')
    ps = request.args.get('ps')
    gewicht = request.args.get('gewicht')
    beschleunigung = request.args.get('beschleunigung')
    baujahr = request.args.get('baujahr')
    prediction_data = [[zylinder,ps,gewicht,beschleunigung,baujahr]]
    result = trained_model.predict(prediction_data).item()
    return {
        'result': result
    }

if __name__ == '__main__':
    app.run()