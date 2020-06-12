#encoding=utf8
import random
import os
import cv2 as cv
import numpy as np

CHAR_SET = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
             ]
index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64,"港":65,"学":66 ,"使":67,"警":68,"澳":70,"挂":71,"军":72,"北":73,"南":74,
         "沈":75,"兰":76,"成":77,"济":78,"海":79,"民":80,"航":81,"空":82}
CHARS_DICT = {char:i for i, char in enumerate(CHAR_SET)}
#===================================#
# 一些全局变量
#===================================#
# 新能源车牌共有8位，其余一般车牌只有7位
WORD_LEN = 7
# 车牌字符种类数
WORD_CLASS = len(CHAR_SET)
# 图片的高
IMG_HEIGHT = 48
# 图片的宽
IMG_WIDTH = 164
# 图片通道数
IMG_CHANNELS = 3
# 训练集路径
#TRAIN_IMG_PATH = './train_plate/'
# 测试集路径
#TEST_IMG_PATH = './test_plate/'
TRAIN_IMG_PATH = 'E:/CCPD/old2/'

# 测试集路径
TEST_IMG_PATH = 'E:/CCPD/test/'
#TEST_IMG_PATH = 'G:/gra_project/About_pro/license-plate-generator-master/WINDKISS_TEST/single_blue/'
# 日志路径
LOG_DIR = './log2/'
# 模型存放路径
MODEL_DIR = './model_data/'
# 模型权重存放路径
#MODEL_WEIGHT_DIR = './model_data/CRNN_weight.h5'
MODEL_WEIGHT_DIR = './model_data/ocr_plate_all_gru.h5'

# 批次内训练图片数量
BATCH_SIZE = 16
# 迭代次数
EPOCHS = 500

#=============================================
# 训练集相关
#=============================================
# 训练集文件具体路径
train_filenames = []
# 训练集标签
train_labels = []
for file_name in os.listdir(TRAIN_IMG_PATH):
    train_filenames.append(os.path.join(TRAIN_IMG_PATH, file_name))
    train_labels.append(file_name[:-4])
# 训练集集数量
TRAIN_SIZE = len(train_labels)
print("TRAIN_SIZE:  ",TRAIN_SIZE)
#=============================================
# 测试集相关
#=============================================
# 测试集文件具体路径
test_filenames = []
# 测试集标签
test_labels = []
for file_name in os.listdir(TEST_IMG_PATH):
    test_filenames.append(os.path.join(TEST_IMG_PATH, file_name))
    test_labels.append(file_name[:-4])
# 测试集数量
TEST_SIZE = len(test_labels)
print("TEST_SIZE:  ",TEST_SIZE)
#===================================#
# 一些函数
#===================================#

# 生成训练\测试数据
def get_img(filename):
    image = []
    for name in filename:
        # img = cv2.imread(name, 0)
        # 由于图片名中有中文
        print("====文件路径:",name)
        # 读入的是灰度图
        img = cv.imdecode(np.fromfile(name, dtype=np.uint8), -1)
        #cv.imshow('t',img)
        # cv2.waitKey()
        img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        #img = cv.transpose(img, (IMG_HEIGHT, IMG_WIDTH))
        img = img.transpose(1, 0, 2)
        #cv.imshow('t2', img)
        #img = cv.flip(img, 1)
        #cv.imshow('t3', img)
        #cv.waitKey()
        img = img / 255.

        image.append(img)
    image = np.array(image)
    # print("===reshape之前:",np.shape(image))
    image = np.reshape(image, (-1, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    # print("===reshape之后:", np.shape(image))
    return image

def gen_train_batch():
    steps = TRAIN_SIZE // BATCH_SIZE
    filenames = np.array(train_filenames)
    label = np.array(train_labels)
    #print("====labels的shape",np.shape(label))
    while True:
        # 打乱训练集
        index = np.random.permutation(TRAIN_SIZE)
        #print("len(labels): ",len(label))
        #print("shuffle_idx:为\n", shuffle_idx)
        filenmaes_shullfled = filenames[index]
        labels_shullfled = label[index]

        for i in range(steps):
            labels = np.ones([BATCH_SIZE, WORD_LEN], dtype=np.int64) * (len(CHAR_SET))
            # print("--=====--",i)
            input_length = [16] * BATCH_SIZE
            label_length = [WORD_LEN] * BATCH_SIZE
            input_length = np.array(input_length)
            label_length = np.array(label_length)

            # 每次读取BATCH_SIZE份数据
            filename_batch = filenmaes_shullfled[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            label_batch = labels_shullfled[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            # 通过图片路径读入图片
            X_data = get_img(filename_batch)

            for j, word in enumerate(label_batch):
                for p, c in enumerate(word):
                    # print("///p",p,"  ",c,"===",int(CHAR_SET.find(c)))
                    labels[j, p] = CHAR_SET.index(c)
                    # print("labels[j, p]:===",labels[j, p])

                cv.imshow('rr',X_data[j])
                print("====", labels[j,:])
                cv.waitKey()
            inputs = {'the_input': X_data, 'the_labels': labels, 'input_length': input_length,
                      'label_length': label_length}
            outputs = {'ctc': labels}

            yield (inputs, outputs)

def gen_test_batch():
    steps = TEST_SIZE // BATCH_SIZE
    filenames = np.array(test_filenames)
    label = np.array(test_labels)
    # print("====raw_labels的shape",np.shape(raw_labels))
    while True:
        # 打乱训练集
        index = np.random.permutation(TEST_SIZE)
        filenmaes_shullfled = filenames[index]
        labels_shullfled = label[index]

        for i in range(steps):
            labels = np.ones([BATCH_SIZE, WORD_LEN], dtype=np.int64) * (len(CHAR_SET))
            # print("--=====--",i)
            input_length = [16] * BATCH_SIZE
            label_length = [WORD_LEN] * BATCH_SIZE
            input_length = np.array(input_length)
            label_length = np.array(label_length)

            # 每次读取BATCH_SIZE份数据
            filename_batch = filenmaes_shullfled[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            label_batch = labels_shullfled[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            # 通过图片路径读入图片
            X_data = get_img(filename_batch)

            for j, word in enumerate(label_batch):
                for p, c in enumerate(word):
                    # print("///p",p,"  ",c,"===",int(CHAR_SET.find(c)))
                    labels[j, p] = CHAR_SET.index(c)
                    # print("rryryry===",labels[j, p])
                # cv2.imshow('rr',X_data[j])
                # print("哈哈哈*****////====", labels[j,:])
                # cv2.waitKey()
            inputs = {'the_input': X_data, 'the_labels': labels, 'input_length': input_length,
                      'label_length': label_length}
            outputs = {'ctc': labels}

            yield (inputs, outputs)

