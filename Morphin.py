#Importing
import numpy as np
import os
from skimage import transform, io
import imageio
import matplotlib.pyplot as plt

def morpher(x,y,alpha, ind):
    # dist_1 = (alpha*y + (1-alpha)*x)*ind + (alpha**5*y + (1-alpha)**5*x)*(1-ind)
    # dist_1 = alpha*x + (1-alpha)*y
    dist_1 = y**alpha*x**(1-alpha)
    return dist_1


def sigmoid(x):
    return x*(x>=0)


read_data = np.loadtxt('DataLabels_in_list_format.txt')
l = np.unique(read_data[:,-1])

n = 4

for i in range(n):
    vars()['fshift_{}'.format(i)] = np.fft.fftshift(np.fft.fft2(np.reshape(read_data[read_data[:, -1] == l[i]][0, :-1], (42, 42))))
    vars()['img_back_{}'.format(i)] = np.abs(np.fft.ifft2(np.fft.ifftshift(vars()['fshift_{}'.format(i)])))
    vars()['magnitude_spectrum_{}'.format(i)] = 20 * np.log(np.abs(vars()['fshift_{}'.format(i)]))


fig, axes = plt.subplots(3,4, sharey=True)
for i in range(n):
    axes[0,i].imshow(np.reshape(read_data[read_data[:,-1] == l[i]][0,:-1],(42,42)),cmap = 'gray')
    axes[1,i].imshow(np.abs(vars()['magnitude_spectrum_{}'.format(i)]), cmap='gray')
    axes[2,i].imshow(vars()['img_back_{}'.format(i)], cmap='gray')

plt.show()


images2 = []
alpha = np.linspace(0,1,100)

pic1 = np.reshape(read_data[read_data[:, -1] == l[0]][0, :-1], (42, 42))
pic2 = np.reshape(read_data[read_data[:, -1] == l[3]][0, :-1], (42, 42))

noise = np.random.normal(size = (42,42))

pic11 = pic1 + noise

fig, ax = plt.subplots(2,3)
ax[0,0].imshow(pic1, cmap = 'gray')
ax[0,1].imshow(noise, cmap = 'gray')
ax[0,2].imshow(pic11, cmap = 'gray')
ax[1,0].imshow( 20 * np.log(np.abs( np.fft.fftshift(np.fft.fft2(pic1)))), cmap = 'gray')
ax[1,1].imshow( 20 * np.log(np.abs( np.fft.fftshift(np.fft.fft2(noise)))), cmap = 'gray')
ax[1,2].imshow( 20 * np.log(np.abs( np.fft.fftshift(np.fft.fft2(pic11)))), cmap = 'gray')
plt.show()

pic_real = np.real(np.fft.fftshift(np.fft.fft2(pic1)))
pic_img = np.imag(np.fft.fftshift(np.fft.fft2(pic1)))

pic_noise_real = np.real(np.fft.fftshift(np.fft.fft2(pic11)))
pic_noise_img = np.imag(np.fft.fftshift(np.fft.fft2(pic11)))

noise_real = np.real(np.fft.fftshift(np.fft.fft2(noise)))
noise_img = np.imag(np.fft.fftshift(np.fft.fft2(noise)))

print(np.max(pic_noise_real - pic_real - noise_real))
print(np.max(pic_noise_img - pic_img - noise_img))

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(pic_noise_real -  pic_real, cmap = 'gray')
ax[0,1].imshow(pic_noise_img - pic_img, cmap = 'gray')
ax[1,0].imshow(noise_real, cmap = 'gray')
ax[1,1].imshow(noise_img, cmap = 'gray')
plt.show()

# for alph in alpha:
#     # dist_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(pic2))*alph + np.fft.fftshift(np.fft.fft2(pic1))*(1-alph)))) # morphing in phase space
#     dist_1 = np.abs(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(pic2)) ** alph *np.fft.fftshift(np.fft.fft2(pic1)) ** (1 - alph))))
#     print(dist_1.shape)
#     images2.append(dist_1.tolist())
#
# images2 = np.array(images2)
# imageio.mimsave('sign_phase.gif', images2, duration = 0.1)

#
# #np.random.shuffle(read_data)
# while read_data[0,-1] == read_data[1,-1]:
#     np.random.shuffle(read_data)
#
# np.savetxt('DataLabels_in_list_format_gifer.txt', read_data)
#
# read_label = read_data[:,-1]
# read_data = read_data[:,:-1]
# print(read_label[0:2])
#
# print(np.unique(read_label))
# conv_label = np.arange(0,len(np.unique(read_label))).tolist()
# label_change_dict = dict(zip(np.unique(read_label), conv_label))
# print(label_change_dict)
# read_label = np.array([label_change_dict[letter] for letter in read_label])
#
#
# print(read_label[0:2])
#
#
# # Load Graph
# sess = tf.Session()
# new_saver = tf.train.import_meta_graph('testing_net_RELU.meta')
# new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#
# graph = tf.get_default_graph()
# x_ = graph.get_tensor_by_name("input:0")
# y_conv = graph.get_tensor_by_name("add_3:0")
# keep_prob = graph.get_tensor_by_name("Placeholder:0")
#
#
#
# # Using Online tool
#
# data = []
#
# IM_DIR = '../Online_Morph'
# for root, diri, item in os.walk(IM_DIR):
#     files = item
#
# files = sorted(files)
#
# for item in files:
#     pic_array = transform.resize(io.imread(IM_DIR + '/' + item, as_gray=True).astype(np.float), (42, 42)).flatten().tolist()
#     data.append(pic_array)
#
# data = np.array(data)
#
#
# tracking = sess.run(y_conv, feed_dict={x_: data, keep_prob: 1.0}).T
#
# for i in range(len(np.unique(read_label))):
#     plt.plot(np.arange(len(files)), sigmoid(tracking[i,:]))
#     plt.legend(('Baustelle','Menschen', 'Kurve', 'Kurve links'))
# plt.show()
#
# # Morphing Images
# Morphed_Images = []
# alpha = np.linspace(0,1,100)
#
# plt.imsave('first_img.png', np.reshape(read_data[0,:],(42,42)), cmap = plt.cm.gray)
# plt.imsave('final_img.png', np.reshape(read_data[1,:],(42,42)), cmap = plt.cm.gray)
#
# ind = np.append(np.ones(42 * 21), np.zeros(42 * 21))
# np.random.shuffle(ind)
#
# for a in alpha:
#     Morphed_Images.append(morpher(read_data[0,:], read_data[1,:], a, ind).tolist())
# morph_label = [read_label[0], read_label[1]]
#
# tracking = sess.run(y_conv, feed_dict={x_: Morphed_Images, keep_prob: 1.0}).T
#
# # print('true image A label: {}'.format(morph_label[0]))
# # print('prediction image A: {}'.format(np.argmax(sigmoid(tracking[:,0]))))
# # print('true image A label: {}'.format(morph_label[1]))
# # print('prediction image B: {}'.format(np.argmax(sigmoid(tracking[:,-1]))))
#
# for i in range(len(np.unique(read_label))):
#     plt.plot(alpha, sigmoid(tracking[i,:]))
#     plt.legend(('Baustelle','Menschen', 'Kurve', 'Kurve links'))
# plt.show()
#
