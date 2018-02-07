import tensorflow as tf
import pickle as pkl
import numpy as np
import gzip
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

with gzip.open("mnist/mnist.pkl.gz", "r") as f:
    train, val, test = pkl.load(f, encoding="latin1")

def one_hot(n, val):
    ret = np.zeros(n)
    ret[val] = 1
    return ret

def data_reshape(data_x, data_y):
    ret_x = [t.reshape(28, 28, 1) for t in data_x]
    ret_y = [one_hot(10, y) for y in data_y]
    return ret_x, ret_y

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

# The model
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y_hat = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y_hat, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

TRAIN_STEPS = 5000

train_x = train[0]
train_y = train[1]
for i in range(TRAIN_STEPS):
    batch_inds = np.random.choice(range(len(train_x)), 100, replace=False)
    batch_X, batch_Y = data_reshape(train_x[batch_inds], train_y[batch_inds])
    _, loss = sess.run([train_step, cross_entropy], feed_dict={X: batch_X, Y: batch_Y, lr: .001})

    print("after training step {}, we have a loss of {}".format(i, loss))

test_X, test_Y = data_reshape(test[0], test[1])
predictions = sess.run(Y_hat, feed_dict={X: test_X, Y: test_Y, lr: .001})
predictions = [np.argmax(p) for p in predictions]

correct = np.sum([p[0] == p[1] for p in zip(predictions, test[1])])
wrong = len(predictions) - correct
mcerror = float(correct) / len(predictions)
print("We classify {} correctly and {} incorrectly giving us an accuracy of {}".format(correct, wrong, mcerror))
