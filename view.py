from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub

from process_image import to_tf_image, to_image_tag_src
from unique_breeds import unique_breeds

app = Flask(__name__)

# temporarily comment out for initial dev.
# model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
#                                      custom_objects={"KerasLayer": hub.KerasLayer})

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

        app.logger.info(tf_image)
    
    return render_template('index.html', image_binary=image_tag_src, tf_image=tf_image)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)