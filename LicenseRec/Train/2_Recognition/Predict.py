import cv2 as cv
import itertools, os, time
import numpy as np
import argparse
from Global import *
import pandas as pd
from keras import backend as K
from CRNN_Model import get_Model_VGG,get_Model
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model


def fastdecode(y_pred):  # 定义置信度计算方法
    results = ""
    confidence = 0.0
    #print("====shape:",np.shape(y_pred))
    y = pd.DataFrame(data=y_pred[0])
    #print(y)
    table_pred = y_pred.reshape(-1, len(CHAR_SET) + 1)
    #print("====shape:", np.shape(table_pred))
    res = table_pred.argmax(axis=1)
    #print("====shape:", np.shape(res))
    y = pd.DataFrame(data=res)
    #print(y)
    for i, one in enumerate(res):  # 计算过程
        if one < len(CHAR_SET) and (i == 0 or (one != res[i - 1])):

            results += CHAR_SET[one]
            confidence += table_pred[i][one]
    confidence /= len(results)  # 结果
    return results, confidence

def recognizeOne(model, src):  # 图像转置，维度进一步处理
    x_tempx = src
    x_temp = cv.resize(x_tempx, (164, 48))
    x_temp = x_temp.transpose(1, 0, 2)
    #cv.imshow('33',x_temp)
    #cv.waitKey()
    y_pred = model.predict(np.array([x_temp]))
    y_pred = y_pred[:, 2:, :]
    return fastdecode(y_pred)
def recognition(image_path, model, label,acc):
    image = []
    # 路径中有中文
    #img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
    #img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    #img = cv.transpose(img, (IMG_HEIGHT, IMG_WIDTH))
    img = []
    img = cv.flip(img, 1)
    img = img / 255.
    image.append(img)


    image = np.array(image)
    image = np.reshape(image, (-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    #model = get_Model(False)


    y_pred = model.predict(image)
    y_pred = y_pred[:, 2:, :]
    #shape = y_pred[:, :, :].shape  # 2:
    #out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,
    #      :WORD_LEN]  # 2:
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, WORD_CLASS+1 )
    res = table_pred.argmax(axis=1)
    for i, one in enumerate(res):  # 计算过程
        if one < WORD_CLASS and (i == 0 or (one != res[i - 1])):
            results += CHAR_SET[one]
            confidence += table_pred[i][one]
    confidence /= len(results)  # 结果
    return results, confidence
    #
    #for i ,plate in enumerate(out):
    #    result = ''
    #    for num in plate:
    #        result += CHAR_SET[num]
    #    print("车牌识别结果：", result)
    #    if result == label:
    #        acc += 1
    #    print("当前acc数：", acc)
    #    print("当前识别率: ",acc/TRAIN_SIZE)

    #return acc

    #print("预测结果：",out)



if __name__ == '__main__':
    #model = get_Model_VGG(False)
    model = model_seq_rec(False)
    plot_model(model, to_file='./CRNN_model_VGG.png', show_shapes=True)
    #i = cv.imread('22.jpg')
    #cv.imshow('ee',i)
    #cv.waitKey()
    model.load_weights('./model_data/ocr_plate_all_gru.h5')
    NUM = 1000
    AC = 0
    for i, filename in enumerate(test_filenames):
        if i < NUM:
            print("第",i+1,"个案例")
            #print("标签为:       ", test_labels[i][0:-2])
            print("标签为:       ", test_labels[i])
            img = cv.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
            #cv.imshow('ee', img)
            results, confidence = recognizeOne(model, img)
            print("===========ocr_plate_all_gru.h5=results:", results)
            acc = 0
            if len(results) == 7:

                for j in range(7):
                    if test_labels[i][j] == results[j]:
                        acc += 1
            if acc == 7:
                AC += 1
            #acc, acc_7, acc_8 = recognition(image_path=filename,model=model,label=test_labels[i][0:-2],acc=acc,acc_7=acc_7,acc_8=acc_8,NUM=TEST_SIZE)
            #acc, acc_7, acc_8 = recognition(image_path=filename, model=model, label=test_labels[i], acc=acc,acc_7=acc_7, acc_8=acc_8, NUM=len(test_filenames))
            print("================AC:",AC)

    #acc = 0
    #for i, filename in enumerate(train_filenames):
    #    if i < 1:
    #        print("第",i,"个案例")
    #        print("标签为:       ", train_labels[i])
    #        results, confidence = recognition(image_path='2002.png',model=model,label=train_labels[i],acc=acc)
    #        print("================results:",results)
