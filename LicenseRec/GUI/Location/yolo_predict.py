import sys
import argparse
from Location.yolo import YOLO, detect_video
from PIL import Image
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path = 'G:/gra_project/GUI/Location/plate/'
def detect_i(yolo,image_path):
    t1 = time.time()
    image = Image.open(image_path)
    name = os.path.basename(image_path)

    print("////  ",name)
    r_image,plate= yolo.detect_image(image)
    time_sum = time.time() - t1
    print(type(plate[0]))
    for i ,p in enumerate(plate):
        save_path = path + name[0:-4] + "_" + str(i) + '.jpg'
        print(save_path)
        plate[i].save(save_path)
    print('time sum:',time_sum)
    save_path = path + name[0:-4] + "_0"  + '.jpg'
    #yolo.close_session()
    return r_image,save_path,len(plate)
# 图片检测
#if __name__ == '__main__':
#    yolo = YOLO()#
#    print("==")
#    r_image,plate,i=detect_i(yolo,'C:/Users/12854/Desktop/test2/7.jpg')
    #image = Image.open('C:/Users/12854/Desktop/test2/7.jpg')
    #r_image, plate = yolo.detect_image(image)
#    print(len(plate))