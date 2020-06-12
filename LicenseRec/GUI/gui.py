import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from Location.yolo_predict import *
from Location.yolo import YOLO, detect_video
from Recognition.Predict import predict
import Recognition.CRNN_Model as CRNN_Model
label_h = 800
label_w = 500
label2_h = 120
label2_w = 360

h=800
w=500

class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.yolo = YOLO()
        self.model = CRNN_Model.get_Model(False)
        # model = CRNN_Model.get_Model_2(False)
        self.model.load_weights('G:/gra_project/GUI/Recognition/model_data/CRNN_weight_plate.h5')

        self.resize(1200, 1200)
        self.setWindowTitle("车牌识别系统")


        self.label = QLabel(self)
        self.label.setText("")
        self.label.setFixedSize(label_w, label_h)
        self.label.move(10, 70)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}"
                                     )
        self.label2=QLabel(self)
        self.label2.setFixedSize(label2_w,label2_h)
        self.label2.move(750,70)
        self.label2.setStyleSheet("QLabel{background:white;}"
                                "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:黑体;}")

        self.label3 = QLabel(self)
        self.label3.setFixedSize(155, 25)
        self.label3.move(750, 570)
        self.label3.setText("识别结果为：")
        self.label3.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:25px;font-weight:bold;font-family:黑体;}")

        self.label4 = QLabel(self)
        self.label4.setFixedSize(label2_w, label2_h)
        self.label4.move(750, 595)
        self.label4.setStyleSheet("QLabel{background:white;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:85px;font-weight:bold;font-family:黑体;}")
        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 30)
        btn.clicked.connect(self.openimage)
    def openimage(self):
        global h
        global w
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(imgName)
        jpg = QtGui.QPixmap(imgName)
        img = jpg.copy()
        #print(jpg.height(),jpg.width())
        img_h = jpg.height()
        img_w = jpg.width()
        print(img_h,img_w)

        # 竖屏
        if img_h >= img_w:
            jpg = jpg.scaled(label_h * img_w // img_h, label_h)
            h = label_h
            w = label_h * img_w // img_h
        #横屏
        elif img_h < img_w:
            jpg = jpg.scaled(label_w, label_w * img_h // img_w)
            h = label_w * img_h // img_w
            w = label_w

        self.label.setFixedSize(w, h)
        self.label.setPixmap(jpg)

       # 开始定位车牌

        r_img,path,l =detect_i(self.yolo,imgName)
        i = QtGui.QPixmap(path)
        img_h =i.height()
        img_w = i.width()
        i = i.scaled(label2_w, label2_w * img_h // img_w)
        h = label2_w * img_h // img_w
        w = label2_w
        self.label2.setFixedSize(w, h)
        self.label2.setPixmap(i)

        # 开始识别车牌
        re = predict(self.model,path)
        self.label4.setText(re)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    #yolo.close_session()
    sys.exit(app.exec_())
