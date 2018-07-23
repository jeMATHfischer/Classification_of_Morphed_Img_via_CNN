import numpy as np
import cv2
import os
import random
from skimage import transform, io
import matplotlib.pyplot as plt

# Building training data from all files in /Images/
FILE_DIR = '../GTSRB/Final_Training/Images/'

dirlist = []
label = []


for root, diri, item in os.walk(FILE_DIR, topdown=False):
    dirlist.append(str(root))

dirlist.remove(dirlist[-1])

data = []
label = []

dirlist = [random.choice(dirlist) for i in range(4)] # for scaling reasons

for dire in dirlist:
    IM_DIR = dire
    i = 0
    for root, diri, item in os.walk(IM_DIR):
        listi = item


        for item in listi:
            if 'csv' in item:
                listi.remove(item)
            else:
                pic_array = cv2.imread(IM_DIR + '/' + item, 0)
                pic_array = cv2.resize(pic_array, (42, 42))
                # img = cv2.imread('wiki.jpg',0)
                # equ = cv2.equalizeHist(pic_array)
                # res = np.hstack((pic_array,equ))
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                res = clahe.apply(pic_array)
                if i == 0:
                    fig, ax = plt.subplots(1,2, sharey= True)
                    ax[0].imshow(pic_array, cmap = 'gray')
                    ax[1].imshow(res, cmap = 'gray')
                    plt.show()
                    i += 1
                pic_array = res.flatten().tolist()
                # pic_array = np.dot(pic_array[..., :3], [0.299, 0.587, 0.114])
                data.append(pic_array)
                if dire[-5:] == '00000':
                    label.append(0)
                else:
                    label.append(float(dire[-5:].replace('000','')))
    print('Done with {}'.format(dire))


read_data = np.append(np.array(data), np.reshape(np.array(label),(-1,1)),axis = 1)
np.savetxt('DataLabels_in_list_format.txt', read_data)
