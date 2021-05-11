import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from scipy import ndimage
import numpy as np
import random
from PIL import Image

com_path = "../dataset2/"


def getimg(imges):
    temp = []
    for ig in imges:
        temp.append(ig)
    return temp


def getdata(batch, path):
    newpath = com_path + path + '/cover'
    newpath2 = com_path + path + '/stego'
    fileList = os.listdir(newpath)
    imges = []
    lables = []

    bathimg = []
    bathlab = []
    for file in fileList:
        if path == 'train':
            a = random.uniform(0, 1)
            if a>0.8:
                continue
        if (os.path.isfile(newpath + '/' + file)):
            lena = Image.open(newpath + '/' + file)
            imges.append(np.array(lena))
            lables.append(0)
            lena1 = Image.open(newpath2 + '/' + file)

            if path == 'train':
                imges.append(np.array(lena1))
                lables.append(1)
                imges.append(np.array(lena.transpose(Image.FLIP_LEFT_RIGHT)))
                imges.append(np.array(lena1.transpose(Image.FLIP_LEFT_RIGHT)))
                imges.append(np.array(lena.transpose(Image.ROTATE_270)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_270)))
                imges.append(np.array(lena.transpose(Image.ROTATE_180)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_180)))
                imges.append(np.array(lena.transpose(Image.ROTATE_90)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_90)))
                lables.append(0)
                lables.append(1)
                lables.append(0)
                lables.append(1)
                lables.append(0)
                lables.append(1)
                lables.append(0)
                lables.append(1)

            if path == 'test' or path=='valid':
                imges.append(np.array(lena.transpose(Image.FLIP_LEFT_RIGHT)))
                imges.append(np.array(lena.transpose(Image.ROTATE_270)))
                imges.append(np.array(lena.transpose(Image.ROTATE_180)))
                imges.append(np.array(lena.transpose(Image.ROTATE_90)))
                imges.append(np.array(lena1.transpose(Image.FLIP_LEFT_RIGHT)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_270)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_180)))
                imges.append(np.array(lena1.transpose(Image.ROTATE_90)))
                lables.append(0)
                lables.append(0)
                lables.append(0)
                lables.append(0)
                lables.append(1)
                lables.append(1)
                lables.append(1)
                lables.append(1)
                imges.append(np.array(lena1))
                lables.append(1)

            if len(lables) == batch:
                bathimg.append(getimg(imges))
                bathlab.append(getimg(lables))
                imges = []
                lables = []
    return bathimg, bathlab
