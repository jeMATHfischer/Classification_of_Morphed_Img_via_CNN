"""
Created on Fri Jun 15 08:34:55 2018

@author: jens
"""

#Importing
import numpy as np
import os
from skimage import transform, io
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# tensorflow functions -------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x,a):
  return tf.nn.max_pool(x, ksize=[1, a, a, 1],
                        strides=[1, a, a, 1], padding='SAME')


# data wrangling functions ---------------------------------
def split_train_test(X,y,split):
    shuffler = np.append(X,np.reshape(y, (-1,1)), axis = 1)
    np.random.shuffle(shuffler)
    ratio = int(split*len(y)) + 1
    X_train = shuffler[:ratio, :-1]
    y_train = shuffler[:ratio, -1]
    X_test  = shuffler[ratio:, :-1]
    y_test = shuffler[ratio:, -1]
    return X_train, y_train, X_test, y_test

def label_hot_encode(y):
    hot_encode = np.ones((len(y), len(np.unique(y))))*np.unique(y)
    for i in range(len(y)):
        hot_encode[i,:] = (hot_encode[i,:] == y[i])*1
    return hot_encode



read_data = np.loadtxt('DataLabels_in_list_format.txt')

training_data, training_label, test_data, test_label = split_train_test(read_data[:, :-1], read_data[:, -1], 0.8)

training_label = label_hot_encode(training_label)
test_label = label_hot_encode(test_label)

np.savetxt('Training_data_HotEncode.txt', np.append(training_data,training_label, axis = 1))
np.savetxt('Test_data_HotEncode.txt', np.append(test_data, test_label, axis = 1))

TrainImBatch, TrainLabBatch = tf.train.shuffle_batch([training_data, training_label], batch_size=50, enqueue_many=True,
                                                     capacity=2000,
                                                     min_after_dequeue=1000)

TestImBatch, TestLabBatch = tf.train.shuffle_batch([test_data, test_label], batch_size=50, enqueue_many=True,
                                                   capacity=2000,
                                                   min_after_dequeue=1000)

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

# Setting Dimensions
dim_label = test_label.shape[1]
dim_pic = training_data.shape[1]

# Inputs
x_ = tf.placeholder(tf.float32, shape=[None, dim_pic], name = 'input')
x_image = tf.reshape(x_, [-1, 42, 42, 1])
y_ = tf.placeholder(tf.float32, shape=[None, dim_label], name = "true_labels")


# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool(h_conv1, 2)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 3)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])  # a*a*64 what is a in my case?
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, dim_label])
b_fc2 = bias_variable([dim_label])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

for i in range(2000):
    if i%50 == 0:
        print(i)
    x_batch, y_batch = sess.run([TrainImBatch, TrainLabBatch])
    sess.run(train_step, feed_dict={x_: x_batch, y_: y_batch, keep_prob: 0.5})
    if i % 100 == 0:
        x_test_batch, y_test_batch = sess.run([TestImBatch, TestLabBatch])
        train_accuracy = accuracy.eval(feed_dict={x_: x_test_batch, y_: y_test_batch, keep_prob: 1.0})
        print('step {}, training accuracy {}'.format(i, train_accuracy))
        print(keep_prob.name)
print('test accuracy {}'.format(accuracy.eval(feed_dict={x_: test_data, y_: test_label, keep_prob: 1.0})))

coord.request_stop()
coord.join(threads)
saver.save(sess, './RELU_net/testing_net_RELU')

