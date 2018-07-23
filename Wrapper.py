#Importing
import numpy as np
import os
import imageio
from skimage import transform, io
import random
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt

def morpher(x,y,alpha, ind):
    dist_1 = np.reshape((alpha*y.flatten() + (1-alpha)*x.flatten())*ind + (alpha**5*y.flatten() + (1-alpha)**5*x.flatten())*(1-ind), (x.shape[0], x.shape[1]))
    # dist_1 = (1-alpha)*x + alpha*y
    # dist_1 = y**alpha*x**(1-alpha)
    return dist_1

def phase_morpher(pic1,pic2,alph, ind):
    dist1 = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(pic2)) ** alph *np.fft.fftshift(np.fft.fft2(pic1)) ** (1 - alph))))
    return dist1

def sigmoid(x):
    return x*(x>=0)


read_data = np.loadtxt('Test_data_HotEncode.txt')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=True)
ax1.imshow(np.reshape(read_data[np.dot(read_data[:,-4:], np.arange(0,4)) == 0][0,:-4],(42,42)),cmap = 'gray')
ax1.set_title(0)
ax2.imshow(np.reshape(read_data[np.dot(read_data[:,-4:], np.arange(0,4)) == 1][0,:-4],(42,42)),cmap = 'gray')
ax2.set_title(1)
ax3.imshow(np.reshape(read_data[np.dot(read_data[:,-4:], np.arange(0,4)) == 2][0,:-4],(42,42)),cmap = 'gray')
ax3.set_title(2)
ax4.imshow(np.reshape(read_data[np.dot(read_data[:,-4:], np.arange(0,4)) == 3][0,:-4],(42,42)),cmap = 'gray')
ax4.set_title(3)
plt.show()

np.random.shuffle(read_data)
while np.dot(read_data[0,-4:], np.arange(0,4)) == np.dot(read_data[1,-4:], np.arange(0,4)):
    np.random.shuffle(read_data)

fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
ax1.imshow(np.reshape(read_data[0,:-4],(42,42)),cmap = 'gray')
ax2.imshow(np.reshape(read_data[1,:-4],(42,42)),cmap = 'gray')
plt.show()

np.savetxt('DataLabels_in_list_format_gifer.txt', read_data)

# Load Graph
sess = tf.Session()
new_saver = tf.train.import_meta_graph('testing_net_RELU.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./RELU_net/'))

graph = tf.get_default_graph()
x_ = graph.get_tensor_by_name("input:0")
y_conv = graph.get_tensor_by_name("add_3:0")
# hidden_1 =
# hidden_2 =
# hidden_3 =
keep_prob = graph.get_tensor_by_name("Placeholder:0")

# Morphing Images and save to gif
Morphed_Images = []
gif_pics = []
alpha = np.linspace(0,1,100)

plt.imsave('first_img.png', np.reshape(read_data[0,:-4],(42,42)), cmap = plt.cm.gray)
plt.imsave('final_img.png', np.reshape(read_data[1,:-4],(42,42)), cmap = plt.cm.gray)

ind = np.append(np.ones(42 * 21), np.zeros(42 * 21))
np.random.shuffle(ind)

for a in alpha:
    Morphed_Images.append(morpher(np.reshape(read_data[0,:-4],(42,42)), np.reshape(read_data[1,:-4],(42,42)), a, ind).flatten().tolist())
    gif_pics.append(morpher(np.reshape(read_data[0,:-4],(42,42)), np.reshape(read_data[1,:-4],(42,42)), a, ind))

images2 = np.array(gif_pics)
imageio.mimsave('sign_phase.gif', images2, duration = 0.1)

morph_label = [np.dot(read_data[0,-4:], np.arange(0,4)), np.dot(read_data[1,-4:], np.arange(0,4))]

# Run net
tracking = sess.run(tf.nn.softmax(y_conv), feed_dict={x_: Morphed_Images, keep_prob: 1.0}).T

print('true image A label: {}'.format(morph_label[0]))
print('prediction image A: {}'.format(np.argmax(tracking[:,0])))
print('true image A label: {}'.format(morph_label[1]))
print('prediction image B: {}'.format(np.argmax(tracking[:,-1])))


fig, (ax1,ax2) = plt.subplots(2,1,sharey=True)
for i in range(4):
    ax1.plot(alpha, sigmoid(tracking[i,:]))
    ax2.plot(alpha, sigmoid(tracking[i,:]))
ax1.legend(('0', '1', '2', '3'))
ax2.legend(('0', '1', '2', '3'))

plt.show()


wrongs = []
for i in range(read_data.shape[0]):
    true_label = np.dot(read_data[i, -4:], np.arange(0, 4))
    test_label = np.dot(np.arange(0, 4), sess.run(tf.nn.softmax(y_conv), feed_dict={x_: np.reshape(read_data[i,:-4],(1,-1)), keep_prob: 1.0}).T)
    if true_label != test_label:
        wrongs.append(i)

print(wrongs)

fig, ax = plt.subplots(1,len(wrongs))
for i in range(len(wrongs)):
    ax[i].imshow(np.reshape(read_data[i,:-4],(42,42)),cmap = 'gray')
plt.show()

wrongs = np.unique((np.triu(np.reshape(np.array(list(itertools.product(wrongs, wrongs)), dtype = 'int,int'), (len(wrongs), -1)), k = 1)).flatten())
# print(wrongs)
output = set()
for x in wrongs:
    output.add(tuple(x))
wrongs = list(output)
del(output)
m = 0

plotter = int((len(wrongs)-1)/2)
fig, ax = plt.subplots(1,plotter,sharey=True)
for pair in wrongs[:plotter]:
    if pair[0] != 0 or pair[1] != 0:
        vars()["Morphed_Images_{}".format(m)] = []
        for a in alpha:
            vars()["Morphed_Images_{}".format(m)].append(morpher(np.reshape(read_data[pair[0], :-4], (42, 42)), np.reshape(read_data[pair[1], :-4], (42, 42)), a,ind).flatten().tolist())
        vars()["tracking_{}".format(m)] = sess.run(tf.nn.softmax(y_conv), feed_dict={x_: vars()["Morphed_Images_{}".format(m)], keep_prob: 1.0}).T
        for j in range(4):
            ax[m].plot(alpha, sigmoid(vars()["tracking_{}".format(m)][j,:]))
            ax[m].set_xlabel('alpha value')
            ax[m].set_ylabel('Softmax guess')
        ax[m].legend(('0', '1', '2', '3'))
        m += 1
plt.show()

# Examining the filters
#
# def getActivations(layer,stimuli):
#     units = sess.run(layer,feed_dict={x: stimuli,keep_prob:1.0})
#     plotNNFilter(units)
#
# def plotNNFilter(units):
#     filters = units.shape[3]
#     plt.figure(1, figsize=(20,20))
#     n_columns = 6
#     n_rows = math.ceil(filters / n_columns) + 1
#     for i in range(filters):
#         plt.subplot(n_rows, n_columns, i+1)
#         plt.title('Filter ' + str(i))
#         plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
#
#
# getActivations(hidden_1,read_data[0,:-4])
#
