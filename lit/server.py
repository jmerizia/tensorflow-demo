from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
from flask import Flask, send_file, jsonify, request
app = Flask(__name__)
from PIL import ImageEnhance, Image
import base64
import numpy as np

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


sess = tf.InteractiveSession()

saver = tf.train.Saver()
saver.restore(sess, "./conv_model/model.ckpt")

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

    prediction = tf.argmax(y_conv, 1)

    flat_image_array = image_array.flatten()
    flat_image_array = flat_image_array[np.newaxis, ...]
    guess = prediction.eval({
        x: flat_image_array,
        keep_prob: 0.5
        })[0]

    ##################### /\
    ## Your Code ABOVE ## ||
    ##################### ||

    return jsonify({"guess": str(guess)})

@app.route('/')
def root():
    return send_file('../www/index.html')

if __name__ == '__main__':
    app.run()
