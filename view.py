from flask import Flask, render_template
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# temporarily comment out for initial dev.
# model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
#                                      custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)