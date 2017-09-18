import tensorflow as tf
import pandas as pd
import flask as fl

hello_message = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello_message))

sess.close()
