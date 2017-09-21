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

# Set neural net variables

# THE COOLEST PART

# Question: What are the preceeding tensors filled with?

# Create a session

# Initialize all variables

# Restore saved model from file

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

    # Make the image black and white

    # Increase the brightness of the image by 2
    
    # Increase the contrast of the image by 3
    
    # Resize the image to 28 by 28

    # Convert the image to an array and normalize it

    # Set up the model prediction

    # Flatten the image
    
    # Question: Why is flattening the image important?

    # Add a dimension to the array

    # Evaluate the prediction, passing in the values for x
    guess = 42

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
