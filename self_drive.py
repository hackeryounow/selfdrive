import numpy as np
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2
import os.path
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from tensorflow.python.keras import callbacks
import math
from matplotlib import pyplot
import tensorboard
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


def image_transformation(img_address, degree, data_dir):
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(data_dir + img_address)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)
    return img, degree


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    """
    随机从左、中、右图像中选择一张图像，并相应调整转动角度
    :param img_address: 中间图像的文件路径
    :param degree: 中间图像对应的方向转动角度
    :param degree_corr: 方向盘转动角度调整的值
    :return:
    """
    swap = np.random.choice(['L', 'R', 'C'])
    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(math.tan(degree) + degree_corr)
        return img_address, corrected_label
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(math.tan(degree) - degree_corr)
        return img_address, corrected_label
    else:
        return img_address, degree


def discard_zero_steering(degrees, rate):
    """
    从角度为零的index中随机选择部分index返回
    :param degrees: 输入角度值
    :param rate: 丢弃率，如果为0.8，80%的会被返回
    :return:
    """
    steering_zero_idx = np.where(degrees == 0)
    steering_zero_idx = steering_zero_idx[0]
    size_label = int(len(steering_zero_idx) * rate)
    return np.random.choice(steering_zero_idx, size=size_label, replace=False)


def random_brightness(img, degree):
    """
    随机调整图像的亮度，调整强度等于0.1（变黑）和1（无变化）之间
    :param img: 输入图像
    :param degree: 输入图像对于的转动角度
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    return rgb, degree


def horizontal_flip(img, degree):
    """
    按照50%的概率水平翻转图像
    :param img: 输入图像
    :param degree: 输入图像对于的转动角度
    :return:
    """
    choice = np.random.choice([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree

    return img, degree


def get_model(shape, keep_prob=0.5):
    """
    预测方向盘角度：以图像为输入，预测方向盘转动角度
    :param shape: 输入图像尺寸，例如（128,128,3）
    :return:
    """
    # 模型需要修改
    model = Sequential()
    model.add(Conv2D(8, (5, 5), strides=(2, 2), padding='valid', activation='relu', input_shape=shape))
    model.add(Conv2D(8, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(16, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # model = Sequential()
    # model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='elu', input_shape=shape))
    # model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    # model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='elu'))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='elu'))
    # model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='elu'))
    # model.add(Dropout(keep_prob))
    # model.add(Flatten())
    # model.add(Dense(1164, activation='elu'))
    # model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dense(1, activation='linear'))
    # compile需要指定优化器,loss 指明是分类问题、回归问题或者其他问题

    sgd = SGD(learning_rate=0.001)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    return model


# def image_transformation(img_address, label, data_dir):
#     """
#     读入数据，可作一定的变换
#     :param img_address:
#     :param label:
#     :param data_dir:
#     :return:
#     """
#     img = cv2.imread(data_dir + img_address)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img, label


def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', discard_rate=0.65):
    """
    产生处理的数据的generator
    :param x: 文件路径list
    :param y: 方向盘角度
    :param batch_size: 批处理大小
    :param shape: 输入图像尺寸（高，宽，通道）
    :param training: True 产生训练数据，False 产生validation数据
    :param data_dir: 数据命令
    :param discard_rate: 随机丢弃角度为0的数据
    :return:
       training=True 返回X, Y
       training=False 返回X
    """
    if training:
        x, y = shuffle(x, y)
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else:
        new_x = x
        new_y = y

    offset = 0
    while True:
        # *shape 将（1,2,3）换为1,2,3
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 图像切割 shape为网络可接受的输入
            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5
            Y[example] = img_steering

            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        yield X, Y

        offset = offset + batch_size


if __name__ == '__main__':
    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # 去掉文件第一行
    # log = log[1:, :]

    # 判断图像文件数量是否等于csv
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 64
    nb_epoch = 10000

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)

    x_, y_ = shuffle(x_, y_)

    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=13)
    print("batch size: {}".format(batch_size))
    print('Train set size:{} | Validation set size:{}'.format(len(X_train), len(X_val)))

    step_per_epoch = 1000
    # 使得validation 数据量大小为batch_size的整数倍
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape)

    print(model.summary())

    # 根据validation loss 保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')
    # 如果训练持续没有validation loss的提升，提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=59,
                                         verbose=0, mode='auto')
    tbCallBack = callbacks.TensorBoard(log_dir='./Graph', write_graph=True, write_images=True)

    callbacks_list = [early_stop, save_best, tbCallBack]
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, (128, 128, 3), training=True),
                                  steps_per_epoch=step_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)
    # 正则化？？？ Dense

    # 保存history
    with open('./trainHistory.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.jpg')

    # 保存模型
    with open('model.json', 'w') as f:
        f.write(model.to_json())

    model.save('model.h5')
    print('Done!')
