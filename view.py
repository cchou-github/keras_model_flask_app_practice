from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from process_image import to_tf_image, to_image_tag_src
from unique_breeds import unique_breeds

app = Flask(__name__)

# temporarily comment out for initial dev.
model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
                                     custom_objects={"KerasLayer": hub.KerasLayer})

# Define the batch size, 32 is a good start
BATCH_SIZE = 32
# Create a function to turn data into batches
def create_data_batches(tf_images, batch_size=BATCH_SIZE, valid_data=False):
    """
    Create batches of data out of image (X) and label (y) pairs.
    Shuffle the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as imput (no labels).
    """
    # If the data is a test dataset, we probably dont have labels
    print("Create test data batches...")
    data = tf.data.Dataset.from_tensor_slices(tf_images) # Only filepaths (no labels)
    data_batch = data.batch(BATCH_SIZE)

    return data_batch

@app.route('/', methods=['GET', 'POST'])
def home():
    image_tag_src = None
    tf_image = None
    app.logger.info(unique_breeds())

    if request.method == 'POST':
        file = request.files['file']
        file_data = file.stream.read()

        image_tag_src = to_image_tag_src(file.content_type, file_data)

        tf_image = to_tf_image(file_data)
        data_batches = create_data_batches([tf_image])
        app.logger.info(data_batches)

        predictions = model.predict(data_batches, verbose=1)

        # First prediction
        index = 0
        app.logger.info(predictions[index])
        app.logger.info(f"Max value(probability of predition): {np.max(predictions[index])}")
        app.logger.info(f"Sum: {np.sum(predictions[index])}")
        app.logger.info(f"Max index: {np.argmax(predictions[index])}")
        app.logger.info(f"Predicted label: {unique_breeds()[np.argmax(predictions[index])]}")
    
    return render_template('index.html', image_binary=image_tag_src, tf_image=tf_image)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)