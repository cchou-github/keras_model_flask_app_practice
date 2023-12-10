from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

from process_image import create_tf_image_data_batches, to_image_tag_src, to_tf_image, plot_prediction_to_binary
from unique_breeds import unique_breeds

app = Flask(__name__)

model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
                                     custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/', methods=['GET', 'POST'])
def home():
    image_tag_src = None
    tf_image = None
    predicted_label = None
    max_probs = None
    plot_image_tag_src = None

    app.logger.info(unique_breeds())

    if request.method == 'POST':
        file = request.files['file']
        file_data = file.stream.read()

        image_tag_src = to_image_tag_src(file.content_type, file_data)

        tf_image = to_tf_image(file_data)
        data_batches = create_tf_image_data_batches([tf_image])
        app.logger.info(data_batches)

        predictions = model.predict(data_batches, verbose=1)

        plot_binary = plot_prediction_to_binary(predictions[0], unique_breeds()) # we only have one prediction at a time for now
        plot_image_tag_src = to_image_tag_src("png", plot_binary)

        # First prediction
        index = 0
        app.logger.info(predictions[index])
        app.logger.info(f"Max value(probability of predition): {np.max(predictions[index])}")
        app.logger.info(f"Sum: {np.sum(predictions[index])}")
        app.logger.info(f"Max index: {np.argmax(predictions[index])}")
        app.logger.info(f"Predicted label: {unique_breeds()[np.argmax(predictions[index])]}")

        predicted_label = unique_breeds()[np.argmax(predictions[index])]
        max_probs = round(np.max(predictions[index]) * 100, 2)
    
    return render_template('index.html', image_binary=image_tag_src, tf_image=tf_image, predicted_label=predicted_label, max_probs=max_probs, plot_image_tag_src=plot_image_tag_src)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)