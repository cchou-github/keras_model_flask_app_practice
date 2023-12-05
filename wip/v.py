
from keras.models import load_model
import tensorflow as tf
from flask import Flask, jsonify, request
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['GET'])
def do_prediction():
    
    res= {"key" : "Hello World!"}
    return jsonify(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0')