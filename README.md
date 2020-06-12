# 本项目采用深度学习的方式实现了整个车牌识别算法

gui.py是使用pyqt5简单写的界面，主要是为了演示，后续有时间可

整个车牌识别主要分为车牌检测与车牌字符识别两部分。 
1.车牌检测模块中使用的是YOLOv3网络，训练数据集来自于CCPD 各子数据集https://github.com/detectRecog/CCPD ，最终 MAP(Mean Average Precision)最高达97.42%，检测速率为1.5s/张。该模型能够初步满足对于车牌检测的精度要求。  
![MAP](https://github.com/windkiss5/LicenseRec/blob/master/snipaste_20200612_192601.jpg)  
2.在车牌字符识别模块中使用的是卷积神经网络与循环神经网络结合(CRNN)的网络模型。卷积层部分的作用是对车牌进行特征提取并最终获取车牌的特征序列。循环层部分的作用是对将特征序列按时间步长进行划分后识别该序列，具体模型使用的双向LSTM，结合正序和反序两种输入方式掌握特征序列的信息，以获得更好的识别效果，最终平均识别率在80%以上。

识别率这里没有到90%+的原因我自我感觉可能是CCPD数据集的图片质量确实被压缩的很厉害，还有就是车牌检测的时候，使用更多的图片去训练会更好一点，这样裁剪下来的图片让CRNN去识别就会更准确。

在做本项目的时候主要遇到的问题在于：  
1.之前没有接触过RNN，只对CNN进行了一定的学习，在将CNN与RNN结合的时候花了较长时间去学习LSTM的公式及对应定义的代码。  
2.另外就是CRNN中CTC_loss的学习，这部分确实我感觉是整个项目最难理解的部分，在最初学习的时候，使用了很深的CNN网络(池化层比较多)，导致最后在与RNN结合的那各特征层图片的宽度太小，根本无法满足车牌字符（这里使用7位）计算CTC_loss的最小长度要求，于是出现了识别车牌序列结果普遍不足7位的情况。在计算CTC_loss时，对于7位的车牌，LSTM的输入是每一个像素长度为输入，时间步数为CNN最终特征图宽度，CTC的input_length需要12位长度。结合下图理解，占位符主要是为了区分连续字符合并为1个字符还是说本来就是两个连续字符的情况，所以12位是例如 湘A-A-A-A-A-A这种情况。 
![CTC](https://github.com/windkiss5/LicenseRec/blob/master/snipaste_20200612_192524.jpg)  
关于CTC算法的理解可以参考https://www.jianshu.com/p/0cca89f64987  
3.本项目使用CRNN去做车牌字符识别而不是一般的CNN再接固定数量的全连接层的目的是为了同时实现识别7位的普通车牌以及8位的新能源车牌，但是本次没有实现对8位车牌的识别，原因主要是CCPD数据集并没有新能源车牌的数据，导致在车牌检测时，不能很好的兼容新能源车牌。后续的改进方案主要时：对于CRNN的车牌字符识别，只需要将Global.py中WORD_LEN变更为8，然后使用7位普通车牌与8位新能源车牌混合的数据集开始训练就行，但是前提还是把车牌检测部分的新能源车牌数据集问题先解决。  

车牌检测效果：  
![LOCATION](https://github.com/windkiss5/LicenseRec/blob/master/snipaste_20200612_192633.jpg)  
车牌识别效果：
![REC](https://github.com/windkiss5/LicenseRec/blob/master/snipaste_20200612_194036.jpg)  
