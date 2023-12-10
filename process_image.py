import tensorflow as tf
import base64
import re


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
