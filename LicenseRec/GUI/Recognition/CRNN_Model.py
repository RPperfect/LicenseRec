#encoding=utf8
import Recognition.Global as Global
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Reshape,Lambda
from keras.layers import Input,Dense,Activation,Dropout,BatchNormalization,TimeDistributed,Flatten
from keras.layers.recurrent import GRU
from keras.models import Model
from keras import backend as K
from keras.layers.merge import add, concatenate



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_Model(training):
    if training:
        keeppro = 0.2
    else:
        keeppro = 0
   # width, height, n_len, n_class = 164, 48, 7, len(Global.CHAR_SET) + 1
    rnn_size = 256
    input_tensor = Input((Global.IMG_WIDTH, Global.IMG_HEIGHT, Global.IMG_CHANNELS))  # 输入层：164*48*3的tensor变量
    x = input_tensor
    base_conv = 32
    for i in range(3):  # 卷积神经网络
        x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)  # 最大池化
    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(base_conv)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)  # 双层的双循环神经网络，正向
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(
        x)  # 反向
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(keeppro)(x)  # 正则化
    x = Dense(Global.WORD_CLASS + 1, kernel_initializer='he_normal', activation='softmax')(x)  # 全连接
    #base_model = Model(inputs=input_tensor, outputs=x)
    #base_model.load_weights(model_path)
    #return base_model

    labels = Input(name='the_labels', shape=[Global.WORD_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [x, labels, input_length, label_length])

    if training:
        return Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    else:
        return Model(inputs=[input_tensor], outputs=x)

def get_Model_VGG(training):
    if training:
        keeppro = 0.2
    else:
        keeppro = 0
    input_shape = (Global.IMG_WIDTH, Global.IMG_HEIGHT, Global.IMG_CHANNELS)

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    inner = Conv2D(64, (3, 3), padding='same', name='conv1')(inputs)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)


    conv_shape = inner.get_shape()
    inner = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(inner)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)
    inner = Dropout(keeppro)(inner)
    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1',dropout=keeppro)(inner)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b',dropout=keeppro)(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged = BatchNormalization()(gru1_merged)

    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2',dropout=keeppro)(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b',dropout=keeppro)(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])
    gru2_merged = BatchNormalization()(gru2_merged)

    inner = Dense(len(Global.CHAR_SET) + 1, kernel_initializer='he_normal', name='dense2')(gru2_merged)
    # 防止过拟合
    #inner = Dropout(keeppro)(inner)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[Global.WORD_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')


    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])  # (None, 1)

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    else:
        return Model(inputs=[inputs], outputs=y_pred)


def get_Model_2(training):
    if training:
        keeppro = 0.2
    else:
        keeppro = 0.
    input_shape = (Global.IMG_WIDTH, Global.IMG_HEIGHT, Global.IMG_CHANNELS)

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    inner = Conv2D(32, (3, 3), padding='same', name='conv1',kernel_initializer='he_normal',strides=1)(inputs)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1', padding='same', strides=2)(inner)

    inner = Conv2D(64, (3, 3), padding='same', name='conv3',kernel_initializer='he_normal',strides=1)(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 1), name='max2',padding='same',strides=2)(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal',strides=1)(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 1), name='max3', padding='same', strides=2)(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal', strides=1)(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)
    inner = MaxPooling2D(pool_size=(2, 1), name='max4', padding='same', strides=2)(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv7', kernel_initializer='he_normal', strides=1)(inner)
    inner = Activation('relu')(inner)
    inner = BatchNormalization()(inner)

    conv_shape = inner.get_shape()
    inner = TimeDistributed(Flatten(),name='timedistrib')(inner)
    inner = Dropout(keeppro)(inner)

    gru_1 = GRU(128, return_sequences=True, kernel_initializer='he_normal', name='gru1',dropout=keeppro)(inner)
    gru_1b = GRU(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b',dropout=keeppro)(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged = BatchNormalization()(gru1_merged)

    gru_2 = GRU(128, return_sequences=True, kernel_initializer='he_normal', name='gru2',dropout=keeppro)(gru1_merged)
    gru_2b = GRU(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b',dropout=keeppro)(gru1_merged)
    gru2_merged = concatenate([gru_2, gru_2b])
    gru2_merged = BatchNormalization()(gru2_merged)


    inner = Dense(len(Global.CHAR_SET) + 1, kernel_initializer='he_normal', name='dense2')(gru2_merged)
    # 防止过拟合

    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[Global.WORD_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')


    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])

    if training:
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
    else:
        return Model(inputs=[inputs], outputs=y_pred)

