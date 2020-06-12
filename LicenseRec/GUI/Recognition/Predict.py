import cv2 as cv
import itertools, os, time
import numpy as np
import argparse
import Recognition.CRNN_Model as CRNN_Model
from Recognition.Global import *
from keras import backend as K

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def recognition(image, model):
    x_tempx = image
    x_temp = cv.resize(x_tempx, (IMG_WIDTH, IMG_HEIGHT))
    x_temp = x_temp.transpose(1, 0, 2)
    y_pred = model.predict(np.array([x_temp]))
    y_pred = y_pred[:, 2:, :]
    results = ""
    table_pred = y_pred.reshape(-1, len(CHAR_SET) + 1)
    res = table_pred.argmax(axis=1)
    for i, one in enumerate(res):  # 计算过程
        if one < len(CHAR_SET) and (i == 0 or (one != res[i - 1])):
            results += CHAR_SET[one]
    return results




def predict(model,path):

    #t = []
    #t.append('1')
    #print("===========t:",t,"========type(t): ",type(t))
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    results = recognition(image=img, model=model)

    #i = cv.imread('222.png')
    #cv.imshow('ee',i)
    #cv.waitKey()
    return results

#if __name__ == '__main__':
#    path = 'G:/gra_project/GUI/Location/plate/0_0.jpg'
#    re = predict(path)
#    print(re)