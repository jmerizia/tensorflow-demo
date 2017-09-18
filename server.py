import tensorflow as tf
from flask import Flask, send_file, jsonify
app = Flask(__name__)

a1 = tf.constant([1, 2, 3])
a2 = tf.constant([1, 2, 3])

a1a2 = tf.multiply(a1, a2)

@app.route('/get_data')
def get_data():
    with tf.Session() as sess:
        output = sess.run(a1a2)
        return jsonify(str(output))

@app.route('/')
def root():
    return send_file('www/index.html')

if __name__ == '__main__':
    app.run()
