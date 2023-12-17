from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

from process_image import create_tf_image_data_batches, to_image_tag_src, to_tf_image, plot_prediction_to_binary, get_top_10_pred_dict
from unique_breeds import unique_breeds

app = Flask(__name__)

# load the model when app starts
model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
                                     custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/', methods=['GET', 'POST'])
def home():
    top_10_pred_dict   = None
    image_tag_src      = None
    tf_image           = None
    top_1_pred_label   = None
    top_1_pred_prob    = None
    plot_image_tag_src = None

    if request.method == 'POST':
        # get the uploaded image and render as base64 string in templates
        file = request.files['file']
        file_data = file.stream.read()
        image_tag_src = to_image_tag_src(file_data)

        # create tf_image_data_batches and predict it.
        tf_image = to_tf_image(file_data)
        data_batches = create_tf_image_data_batches([tf_image])
        predictions = model.predict(data_batches, verbose=1)

        # get top 10 predictions
        top_10_pred_dict = get_top_10_pred_dict(predictions[0], unique_breeds())
        top_1_pred_label = next(iter(top_10_pred_dict)) # first key in top_10_pred_dict
        top_1_pred_prob = top_10_pred_dict[top_1_pred_label]

        plot_binary        = plot_prediction_to_binary(top_10_pred_dict) # we only have one prediction at a time for now
        plot_image_tag_src = to_image_tag_src(plot_binary)
    
    return render_template('index.html',
                            image_binary=image_tag_src,
                            tf_image=tf_image,
                            top_1_pred_label=top_1_pred_label,
                            top_1_pred_prob=top_1_pred_prob,
                            top_10_pred_dict=top_10_pred_dict,
                            plot_image_tag_src=plot_image_tag_src)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)