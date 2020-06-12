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
#TRAIN_IMG_PATH = './train_plate_7/'


# 批次内训练图片数量
BATCH_SIZE = 32
# 迭代次数
EPOCHS = 500


