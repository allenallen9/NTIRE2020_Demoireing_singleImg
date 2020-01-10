import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageChops
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import math
import random
import cv2
import time
np.random.seed(8888)

def set_gpu_config(device = 0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[device], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


PATH = os.path.splitext(os.path.basename(__file__))[0] + '/'
if not os.path.exists(PATH):
    os.mkdir(PATH)

def dataNormalize(x):
    return ((x.astype(np.float32)-127.5)/127.5)
    # return ((x.astype(np.float32))/255)
def dataUnNormalize(x):
    return np.clip(x,-1,1)*127.5+127.5
    # return np.clip(x,0,1)*255
def normalizeDataClip(x):
    return np.clip(x,-1,1)
    # return np.clip(x,0,1)
####
def pltLoss(tilte, plt1, plt2):
    plt.plot(history[plt1])
    plt.plot(history[plt2])
    plt.title(tilte)
    plt.ylabel(tilte)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(path + tilte + ".png")
    plt.close()
def plotHistory(hist_list,name):
    hist = np.array(hist_list)
    metric_num = hist.shape[1]
    for i in range(0,metric_num,1):
        metric_name = name[i]
        plt.plot(hist[:,i])
        plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.savefig(PATH + '/' + metric_name + ".png")
        plt.close()
    return
####
def load_data():
    gt_path = 'dataset/TrainSingle/gt/'
    train_path = 'dataset/TrainSingle/input/'
    dataNum = 10000
    trainNum= int(dataNum*0.9)
    imgH = imgW = 128

    # load train data
    x = np.zeros([dataNum, imgH, imgW, 3])
    y = np.zeros([dataNum, imgH, imgW, 3])
    for i in range(dataNum):
        print('\rload data {}/{}'.format(i,dataNum),end='')
        x[i] = np.array(Image.open(train_path + '{:06d}_3.png'.format(i)).convert('RGB')).astype(np.float32)
        y[i] = np.array(Image.open(gt_path + '{:06d}_gt.png'.format(i)).convert('RGB')).astype(np.float32)
    x = dataNormalize(x)
    y = dataNormalize(y)

    idx_list = np.arange(dataNum)
    # np.random.shuffle(idx_list)
    train_x = x[idx_list[:trainNum]]
    train_y = y[idx_list[:trainNum]]
    val_x = x[idx_list[trainNum:]]
    val_y = y[idx_list[trainNum:]]

    return train_x, train_y, val_x, val_y
def data_generator(train_x, train_y, batch_size):
    dataNum = train_x.shape[0]

    while True:
        idx=np.random.randint(dataNum,size=batch_size)
        x = train_x[idx]
        y = train_y[idx]
        # #aug
        # flip_axis = np.random.choice([1,2])
        # x = np.flip(x,axis=flip_axis)
        # y = np.flip(y,axis=flip_axis)
        # rot_num = np.random.randint(4)
        # x = np.rot90(x,rot_num,axes=(1,2))
        # y = np.rot90(y,rot_num,axes=(1,2))

        yield x, y

##
def resBlock(x, filters):

    y = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Add()([x,y])
    y = Activation('elu')(y)
    return y
def resBlock_down(x, filters):
    res = Conv2D(filters, (4, 4), padding='same', strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

    y = Conv2D(filters, (4, 4), padding='same', strides=(2, 2), kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Add()([res,y])
    y = Activation('elu')(y)
    return y
def resBlock_up(x, filters):
    y = x
    y = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = UpSampling2D((2, 2))(y)
    res = y
    y = Conv2D(filters, (4, 4), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    y = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Add()([res, y])
    y = Activation('elu')(y)
    return y
def convBlock(x, filters, ksize):
    y = Conv2D(filters, ksize, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)
    return y
def seBlock(x, filters):
    y = Lambda(lambda x: tf.reduce_mean(x,axis=[1,2],keepdims=True))(x)
    y = convBlock(y, filters, (1, 1))
    y = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = Activation('sigmoid')(y)
    y = Lambda(lambda x: x[0]*x[1])([x, y])
    return y
def build_model():


    filters = [32,64,128,256,256] #128*128 => 4*4
    scale = len(filters)
    feat=[]
    output=[]

    x = Input((128, 128, 3))

    y = x
    y = Conv2D(filters[0], (5, 5), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    y = BatchNormalization()(y)
    y = Activation('elu')(y)

    for i in range(scale):
        y = resBlock_down(y, filters[i])
        feat.append(y)

    for i in range(4):
        y = resBlock(y, filters[-1])
    y = seBlock(y, filters[-1])

    for i in range(len(feat)-1,-1,-1):
        y = Concatenate()([y, feat[i]])
        y = resBlock_up(y, filters[i])

    y = Conv2D(3, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
    # y = Activation('tanh')(y)
    y = Add()([y, x])

    return Model(inputs=[x], outputs=[y])

def build_train_step(model):
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5)

    @tf.function
    def train_step(x, y):
        # set for training updates of SNConv2D and SNDense
        tf.keras.backend.set_learning_phase(True)

        pre = model(x)
        # loss = tf.reduce_mean(tf.abs(pre-y))
        loss = tf.reduce_mean(tf.square(pre-y))

        variable = model.trainable_variables
        gradients = tf.gradients(loss, variable)
        optimizer.apply_gradients(zip(gradients, variable))

        return loss

    return train_step

def trainFunc():
    TRAIN_BATCH_SIZE = 16
    TRAIN_STEP = 1000
    EPOCH = 100

    set_gpu_config(1)
    model = build_model()
    model.summary()
    train_step = build_train_step(model)

    train_x, train_y, val_x, val_y = load_data()
    print('train_size = {}'.format(train_x.shape))
    print('  val_size = {}'.format(val_x.shape))

    data_gen = data_generator(train_x, train_y, TRAIN_BATCH_SIZE)
    # while True:
    #     x, y = next(data_gen)
    #     print(x.shape,y.shape)
    #     plt.subplot(121)
    #     plt.imshow(dataUnNormalize(x[0]).astype(np.uint8))
    #     plt.subplot(122)
    #     plt.imshow(dataUnNormalize(y[0]).astype(np.uint8))
    #     plt.show()

    history_loss = []
    history_val_psnr = []
    best_val = 0
    loss_file = open(PATH + 'loss.txt', mode='w')
    loss_file.close()
    val_file = open(PATH + 'val.txt', mode='w')
    val_file.close()
    for e in range(EPOCH):
        mean_loss = 0
        for s in range(TRAIN_STEP):
            x, y = next(data_gen)
            loss = train_step(x, y)
            mean_loss += loss / TRAIN_STEP
            print("\r[{}/{}] [{}/{}] loss: {}".format(e, EPOCH, s, TRAIN_STEP, loss), end='')

        history_loss.append([mean_loss])
        plotHistory(history_loss,['loss'])

        loss_file = open(PATH + 'loss.txt', mode='a')
        loss_file.write('{} {}\r\n'.format(e, history_loss[e][0]))
        loss_file.close()

        #val
        pre = model.predict(val_x, batch_size=64)
        mse = np.mean(np.reshape(np.square(val_y - pre), -1))
        val_psnr = np.log10(2**2 / mse) * 10

        print('   val_psnr = {}'.format(np.round(val_psnr,2)))

        history_val_psnr.append([val_psnr])
        plotHistory(history_val_psnr, ['val_psnr'])

        val_file = open(PATH + 'val.txt', mode='a')
        if val_psnr>best_val:
            best_val = val_psnr
            val_file.write('{} {}, best\r\n'.format(e, val_psnr))
            model.save_weights(PATH+'model.h5')
        else:
            val_file.write('{} {}\r\n'.format(e, val_psnr))
        val_file.close()
        


if __name__ == '__main__':
    trainFunc() #for train
