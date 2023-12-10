import tensorflow as tf

# Define image size
IMAGE_SIZE = 224
# Create a function for preprocessing images
def process_image(image_binary, image_size=IMAGE_SIZE):
  """
  Takes an image_binary and turns the iamge into a Tensor.
  """
  # Read in an image file
  # image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (R, G, B)
  tf_image = tf.image.decode_jpeg(image_binary, channels=3)
  # Convert the colour channel values from 0-255 to 0-1 values
  tf_image = tf.image.convert_image_dtype(tf_image, tf.float32)
  # Resize the image to our desired value (224, 224)
  tf_image = tf.image.resize(tf_image, size=[IMAGE_SIZE, IMAGE_SIZE])

  return tf_image
