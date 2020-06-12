#encoding=utf8

from keras.optimizers import Adam,SGD,Adadelta,RMSprop
from keras.callbacks import *
import Global
import CRNN_Model
import os
import cv2 as cv
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def train():
    #model = CRNN_Model.get_Model_VGG(training = True)
    basemodel, model = CRNN_Model.model_seq_rec(training=True)
    # 打印输出模型结构图
    print(model.summary())
    plot_model(model, to_file='./CRNN_model_Final.png', show_shapes=True)
    #i = cv.imread('22.jpg')
    #cv.imshow('ee',i)
    #cv.waitKey()
    try:
        model.load_weights(Global.MODEL_WEIGHT_DIR)
        print("正在载入已有模型权重...")
    except:
        print("正在准备重新训练...")
        pass
    checkpoint = ModelCheckpoint(filepath=Global.MODEL_WEIGHT_DIR, save_best_only=True, monitor='val_loss', verbose=1,
                                 mode='min', period=1)
    RL = ReduceLROnPlateau(monitor='val_loss', patience=2)
    TB = TensorBoard(log_dir=Global.LOG_DIR)
    # 早停
    RS = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)

    adam = Adadelta()

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    model.fit_generator(generator=Global.gen_train_batch(),
                        steps_per_epoch=Global.TRAIN_SIZE // Global.BATCH_SIZE,
                        epochs=Global.EPOCHS,
                        callbacks=[checkpoint, TB, RL,RS],
                        validation_data=Global.gen_test_batch(),
                        validation_steps=Global.TEST_SIZE // Global.BATCH_SIZE)


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    KTF.set_session(session)

    # 开始训练
    train()
