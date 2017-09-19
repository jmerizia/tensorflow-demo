import tensorflow as tf
from flask import Flask, send_file, jsonify, request
#import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import ImageEnhance, Image
import numpy as np
import base64
import time
app = Flask(__name__)

##################### ||
## Your Code BELOW ## ||
##################### \/

# Goal: set up lazy loading


##################### /\
## Your Code ABOVE ## ||
##################### ||

@app.route('/estimate', methods=['POST'])
def get_data():
    with tf.Session() as sess:
        encoded_img_data = request.form.get('img_data')
        encoded_img_data = encoded_img_data.replace("@", "+")
        decoded_img_data = base64.b64decode(encoded_img_data)
        g = open("number.png", "wb")
        g.write(decoded_img_data)
        g.close()
        image = Image.open("number.png")
        image = ImageEnhance.Brightness(image).enhance(2.3)
        image = ImageEnhance.Contrast(image).enhance(3)
        image = image.resize((28, 28))

        arr = np.array(image)

        ##################### ||
        ## Your Code BELOW ## ||
        ##################### \/

        # Goal: write the forward propagation algorithm

        #plt.imshow(image)
        #plt.show()

        ##################### /\
        ## Your Code ABOVE ## ||
        ##################### ||

        return jsonify({"guess": 2})

@app.route('/')
def root():
    return send_file('www/index.html')

if __name__ == '__main__':
    app.run()
