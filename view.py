from flask import Flask, render_template, request
import os
import logging

from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub

from process_image import process_image
from unique_breeds import unique_breeds

app = Flask(__name__)

# temporarily comment out for initial dev.
# model = tf.keras.models.load_model("20231118-13041700312647-full-images-mobilenet2-Chou.h5",
#                                      custom_objects={"KerasLayer": hub.KerasLayer})

@app.route('/', methods=['GET', 'POST'])
def home():
    image_binary = None
    tf_image = None
    app.logger.info(unique_breeds())

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
        app.logger.info(tf_image)
    
    return render_template('index.html', image_binary=image_binary, tf_image=tf_image)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)