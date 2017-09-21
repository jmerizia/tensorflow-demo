import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
from flask import Flask, send_file, jsonify, request
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
import numpy as np
import base64
import time
app = Flask(__name__)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y_ * tf.log(y),
    reduction_indices=[1])
)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


##################### ||
## Your Code BELOW ## ||
##################### \/

# Goal: restore saved model from file

saver = tf.train.Saver()

saver.restore(sess, "./model/model.ckpt")

##################### /\
## Your Code ABOVE ## ||
##################### ||

@app.route('/estimate', methods=['POST'])
def get_data():
    encoded_img_data = request.form.get('img_data')
    encoded_img_data = encoded_img_data.replace("@", "+")
    decoded_img_data = base64.b64decode(encoded_img_data)
    g = open("number.png", "wb")
    g.write(decoded_img_data)
    g.close()

    ##################### ||
    ## Your Code BELOW ## ||
    ##################### \/

    image = Image.open("number.png")
    image = ImageEnhance.Color(image).enhance(0.0)
    image = ImageEnhance.Brightness(image).enhance(2)
    image = ImageEnhance.Contrast(image).enhance(3)
    image = image.resize((28, 28))

    image_array = 0.992 - (np.array(image)[:, :, 0] / 255.0)

    prediction = tf.argmax(y, 1)

    flat_image_array = image_array.flatten()
    flat_image_array = flat_image_array[np.newaxis, ...]
    guess = prediction.eval({x: flat_image_array})[0]

    plt.imshow(flat_image_array.reshape((28, -1)))

    plt.show()

    ##################### /\
    ## Your Code ABOVE ## ||
    ##################### ||

    return jsonify({"guess": str(guess)})

@app.route('/')
def root():
    return send_file('www/index.html')

if __name__ == '__main__':
    app.run()
