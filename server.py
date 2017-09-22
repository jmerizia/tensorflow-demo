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

##################### ||
## Your Code BELOW ## ||
##################### \/

# Set x placeholder
x = tf.placeholder(tf.float32, [None, 784])

# Set neural net variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# THE COOLEST PART
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Question: What are the preceeding tensors filled with?

# Create a session
sess = tf.InteractiveSession()

# Initialize all variables
tf.global_variables_initializer().run()

# Restore saved model from file
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

    # Open the image
    image = Image.open("number.png")

    # Make the image black and white
    image = ImageEnhance.Color(image).enhance(0.0)

    # Increase the brightness of the image by 2
    image = ImageEnhance.Brightness(image).enhance(2)
    
    # Increase the contrast of the image by 3
    image = ImageEnhance.Contrast(image).enhance(3)
    
    # Resize the image to 28 by 28
    image = image.resize((28, 28))

    # Convert the image to an array and normalize it
    image_array = 1 - (np.array(image)[:,:,0]/255.0)

    # Set up the model prediction
    prediction = tf.argmax(y, 1)

    # Flatten the image
    flat_image_array = image_array.flatten()
    
    # Question: Why is flattening the image important?

    # Add a dimension to the array
    flat_image_array = flat_image_array[np.newaxis, ...]

    # Evaluate the prediction, passing in the values for x
    guess = prediction.eval({x: flat_image_array})[0]

    #plt.imshow(flat_image_array.reshape((28, -1)))

    #plt.show()

    ##################### /\
    ## Your Code ABOVE ## ||
    ##################### ||

    return jsonify({"guess": str(guess)})

@app.route('/')
def root():
    return send_file('www/index.html')

if __name__ == '__main__':
    app.run()
