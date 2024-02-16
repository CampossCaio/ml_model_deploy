# importar os pacotes necessários
import numpy as np
from flask import Flask, jsonify, request
from joblib import load


# instanciar Flask object
app = Flask(__name__)

# model
model = load('model/model.joblib')

@app.route("/")
def home():
    return { "name": "Caio campos" }

@app.route('/predict', methods=['POST'])
def predict():
    args = request.get_json(force=True)
    input_values = np.asarray(list(args.values())).reshape(1, -1)
    
    predicted = model.predict(input_values)[0]

    return jsonify({'predict': float(predicted)})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)