import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import base64
import re
import io


IMAGE_SIZE = 224 # Define image size. 224 * 224 is required by mobilenet V2
BATCH_SIZE = 32 # Define the batch size, 32 is a good start

def to_tf_image(image_binary, image_size=IMAGE_SIZE):
    """
    Takes an image_binary and turns the image into a Tensor.
    """
    # Turn the jpeg image into numerical Tensor with 3 colour channels (R, G, B)
    tf_image = tf.image.decode_jpeg(image_binary, channels=3)
    # Convert the colour channel values from 0-255 to 0-1 values
    tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
    # Resize the image to our desired value (224, 224)
    tf_image = tf.image.resize(tf_image, size=[IMAGE_SIZE, IMAGE_SIZE])

    return tf_image

def create_tf_image_data_batches(tf_images, batch_size=BATCH_SIZE):
    """
    Create batches of data out of Tensor images.
    """
    # If the data is a test dataset, we probably dont have labels
    data = tf.data.Dataset.from_tensor_slices(tf_images) # Only filepaths (no labels)
    data_batch = data.batch(BATCH_SIZE)

    return data_batch

def to_image_tag_src(file_content_type, file_data):
    """
    Create base64 strings for image tag in HTML
    """
  
    image_tag_src_content_type = ''

    # ファイル形式を取得
    if 'png' in file_content_type:
        image_tag_src_content_type = 'png'
    elif 'jpeg' in file_content_type:
        image_tag_src_content_type = 'jpeg'

    # bytesファイルのデータをbase64にエンコードする
    uploadimage_base64 = base64.b64encode(file_data)
    # base64形式のデータを文字列に変換する。その際に、「b'」と「'」の文字列を除去する
    uploadimage_base64_string = re.sub('b\'|\'', '', str(uploadimage_base64))
    # 「data:image/png;base64,xxxxx」の形式にする
    return f'data:image/{image_tag_src_content_type};base64,{uploadimage_base64_string}'

def get_top_10_pred_dict(pred_prob, unique_breeds):
    # Find the top 10 prediction confidence indexes
    top_10_pred_indexes = pred_prob.argsort()[-10:]
    # Find the top 10 prediction confidence values
    top_10_pred_values = pred_prob[top_10_pred_indexes]
    # Find the top 10 prediction labels
    top_10_pred_labels = unique_breeds[top_10_pred_indexes]

    top_10_pred_dict = {}
    for i in range(0, 10):
      top_10_pred_dict[top_10_pred_labels[i]] = top_10_pred_values[i]

    return top_10_pred_dict


def plot_prediction_to_binary(top_10_pred_dict):
    """
    Plot the top 10 highest prediction confidences
    """
    top_10_pred_labels = list(top_10_pred_dict)
    top_10_pred_values = list(top_10_pred_dict.values())

    # Setup plot
    top_plot = plt.barh(np.arange(len(top_10_pred_labels)),
                      top_10_pred_values,
                      color="orange")
    plt.xlim([0, 1])
    plt.yticks(np.arange(len(top_10_pred_labels)),
              labels=top_10_pred_labels
              )
    # Display values on the bars
    for index, value in enumerate(top_10_pred_values):
        plt.text(value, index, f'{value:.4f}', ha='left', va='center', color="grey")

    
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png', bbox_inches='tight')
    image_binary = image_stream.getvalue()
    image_stream.close()

    return image_binary
