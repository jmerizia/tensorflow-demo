import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))


# from tensorflow.examples.tutorials.mnist import ipnut_data
# mnist = input_data.read_data_set("/tmp/data", one_hot=True)
# 
# learning_rate = 0.1
# num_steps = 500
# batch_size = 128
# display_step = 100
# 
# # Network Parameters
# n_hidden_1 = 256 # 1st layer number of neurons
# n_hidden_2 = 256 # 2nd layer number of neurons
# num_input = 784 # MNIST data input (img shape: 28*28)
# num_classes = 10 # MNIST total classes (0-9 digits)
# 
# X = tf.placeholder("float", [None, num_input])
# Y = tf.placeholder("float", [None, num_classes])
# 
# weights = {
#         'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
#         'h2': tf.Variable(tf.random_normal(n_hidden_1, n_hidden_2)),
#         'out': tf.Variable(tf.random_normal(n_hidden_2, num_classes)) 
#         }
# 
# biases = {
#         'b1': tf.Variable(tf.random_normal(n_hidden_1)),
#         'b2': tf.Variable(tf.random_normal(n_hidden_2)),
#         'out': tf.Variable(tf.random_normal(num_classes))
#         }
# 
# def neural_net(x):
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
#     return out_layer
# 
# logits = neural_net(x)
