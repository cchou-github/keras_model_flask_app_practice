from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub

app = Flask(__name__)

# 120 unique breeds
unique_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

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

# temporarily comment out for initial dev.
# model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
#                                      custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/', methods=['GET', 'POST'])
def home():
    image_binary = None
    tf_image = None

    if request.method == 'POST':
        file = request.files['file']
        file_data = file.stream.read()
        tf_image = process_image(file_data)

        # bytesをbase64にエンコードするライブラリをインポート
        import base64
        import re

        content_type = ''
        # ファイル形式を取得
        if 'png' in file.content_type:
            content_type = 'png'
        elif 'jpeg' in file.content_type:
            content_type = 'jpeg'
 
        # bytesファイルのデータをbase64にエンコードする
        uploadimage_base64 = base64.b64encode(file_data)
        
        # base64形式のデータを文字列に変換する。その際に、「b'」と「'」の文字列を除去する
        uploadimage_base64_string = re.sub('b\'|\'', '', str(uploadimage_base64))
        
        # 「data:image/png;base64,xxxxx」の形式にする
        image_binary = f'data:image/{content_type};base64,{uploadimage_base64_string}'
        app.logger.error(tf_image)
    
    return render_template('index.html', image_binary=image_binary, tf_image=tf_image)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)