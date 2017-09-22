import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set x placeholder
x = tf.placeholder(tf.float32, [None, 784])

# Set neural net variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# THE COOLEST PART: THE MODEL
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Set up a placeholder for the actual y, y_
y_ = tf.placeholder(tf.float32, [None, 10])

# Set up the cost function
cross_entropy = tf.reduce_mean(
        -tf.reduce_mean(y_ * tf.log(y),
        reduction_indices=[1])
        )

# Set up the training algorithm
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# Create a session
sess = tf.InteractiveSession()

# Initialize all variables
tf.global_variables_initializer().run()

# Train the model parameters
for i in range(10000):
    if (i % 1000 == 0):
        print("Iteration: ", i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={
        x: batch_xs,
        y_:batch_ys
        })

# Lazy loader for correct predictions
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Calculate the accuracy of the model
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Print results
print(sess.run(accuracy, feed_dict={
    x: mnist.test.images,
    y_: mnist.test.labels
    }))

# Save model to folder
saver = tf.train.Saver()
saver.save(sess, "./model/model.ckpt")
print("Model saved!")
