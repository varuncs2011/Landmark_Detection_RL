import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage import color



def display_image(image, point):
    fig, ax = plt.subplots()
    print(point)
    #image = color.gray2rgb(image)
    #image = color.rgb2gray(image)

    image = cv2.cvtColor(image.astype("float32"),cv2.COLOR_GRAY2BGR)
    #image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    image[point[0], point[1]] = 255
    print(image.shape)
    ax.imshow(image, cmap='gray_r')
    ax.axis('off')  # clear x- and y-axes
    plt.show()

def display_image1(image):
    fig, ax = plt.subplots()
    image = color.gray2rgb(image)
    print(image)
    ax.imshow(image, cmap='gray_r')
    ax.axis('off')  # clear x- and y-axes
    plt.show()


def make_image(x, y, img):
    words = img.split()
    results = list(map(int, words))
    x_new = np.array(results)
    x1 = x_new.reshape(96, 96)
    landmark1 = [0, 0]
    print(x1.shape)
    print(x1.dtype)
    x1 = x1.astype('uint8')
    print(x1.dtype)
    cv2.imshow('image',x1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if math.isnan(x) or math.isnan(y):
        return x1, landmark1
    else:
        landmark = [int(x), int(y)]
        return x1, landmark


df_r = pd.read_csv("training.csv", skiprows=2000, nrows=30, sep=",",
                   usecols=[20, 21, 30])


for index, row in df_r.iterrows():
    image, point = make_image(row[0],row[1],row[2])
    display_image(image, point)
    crop_img = image[24:74, 24:74]
    point = (point[0]-24, point[1]-24)
    display_image(crop_img, point)
   # display_image1(crop_img)


