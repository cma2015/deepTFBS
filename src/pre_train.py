# -*- coding: utf-8 -*-
# Copyright 2024 Jingjing Zhai, Minggui Song.
# All rights reserved.

"""Represent a collect flnc information.

What's here:

PreTrain positive and negative data sets.
-------------------------------------------

Classes:
    - PreTrain
"""
from logging import getLogger
from src.sys_output import Output
from pathlib import Path
import os,random,sys
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow.keras.metrics
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import numpy as np
from tensorflow.keras.models import Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def conv_block(x):
    conv1 = Convolution1D(filters=256, kernel_size=1, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Convolution1D(filters=128, kernel_size=3, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Convolution1D(filters=256, kernel_size=1, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv4 = Convolution1D(filters=128, kernel_size=3, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    tt = Convolution1D(filters=128, kernel_size=3, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(x)
    tt = BatchNormalization()(tt)
    tt = Activation('relu')(tt)
    out = Add()([conv4, tt])
    out = Activation('relu')(out)
    return out

def identity_block(x):
    conv1 = Convolution1D(filters=256, kernel_size=1, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Convolution1D(filters=128, kernel_size=3, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv3 = Convolution1D(filters=256, kernel_size=1, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv4 = Convolution1D(filters=512, kernel_size=3, strides=1,
                          kernel_initializer='glorot_normal',
                          padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    out = Add()([conv4, x])
    out = Activation('relu')(out)
    return out

def build_model():
    input1 = Input(shape = (1000, 4))
    tfbs = Convolution1D(filters=512, kernel_size=8,
                       kernel_initializer='glorot_normal',
                       strides=1)(input1)
    tfbs = BatchNormalization()(tfbs)
    tfbs = Activation('relu')(tfbs)
    tfbs = identity_block(tfbs)
    tfbs = AveragePooling1D(pool_size=3)(tfbs)

    bilstm = Bidirectional(LSTM(256, return_sequences=True), merge_mode='concat')(tfbs)
    att = Attention()([bilstm, bilstm, bilstm])
    att = GlobalAveragePooling1D()(att)
    out = Dropout(0.5)(att)
    out = Dense(512, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(1, activation='sigmoid')(out)

    model = Model(input1, out)
    return model

logger = getLogger(__name__)  # pylint: disable=invalid-name

class PreTrain(object):
    """
    Attributes:
        - args: Arguments.
        - output: Output info, warning and error.

    """
    def __init__(self, arguments) -> None:
        """Initialize CollectFlncInfo."""
        self.args = arguments
        self.output = Output()
        self.output.info(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')
        logger.debug(
            f'Initializing {self.__class__.__name__}: (args: {arguments}.')
    
    def deepTFBS_data_provider(self) -> None:
        """Provide data for deepTFBS."""
        self.output.info('Starting deepTFBS data loading.')
        logger.debug('Starting deepTFBS data loading.')

        self.TFBSDic = self.args.TFBSDic
        #"/home/malab21/disk2/deepTFBS/1.performance.evaluation/1.samples/10.binary/"
        self.TF = self.args.TF
        loaded = np.load(f'{self.TFBSDic}/{self.TF}/train.npz')
        train_input = loaded['a']
        self.train_input = np.transpose(train_input, [0,2,1])

        loaded = np.load(f'{self.TFBSDic}/{self.TF}/validation.npz')
        val_input = loaded['a']
        self.val_input = np.transpose(val_input, [0,2,1])

        self.train_label = np.loadtxt(f'{self.TFBSDic}/{self.TF}/train_label.txt', delimiter = '\t', dtype = 'int')
        self.val_label = np.loadtxt(f'{self.TFBSDic}/{self.TF}/validation_label.txt', delimiter = '\t', dtype = 'int')

        self.output.info('Completed deepTFBS data loading.')
        logger.debug('Completed deepTFBS data loading.')

    def pretraining(self) -> None:
        """Pretraining."""
        self.output.info('Starting Pretraining.')
        logger.debug('Starting Pretraining.')
        tensorflow.keras.metrics.f1 = f1
        model = build_model()
        batch_size = self.args.batch_size
        n_epochs = self.args.n_epochs
        lr = self.args.lr_init
        epsilon = self.args.epsilon
        early_stopping = EarlyStopping(monitor='val_loss',patience=10)
        Adam = Adam(lr = lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = epsilon)
        model.compile(optimizer = Adam, loss = binary_crossentropy, metrics = [f1])
        checkpoint = ModelCheckpoint(f'{self.args.resDic}/{self.args.TF}_best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
        with tf.device('/device:GPU:0'):
            history_callback = model.fit(x = self.train_input, y = self.train_label,
                validation_data=(self.val_input, self.val_label),
                batch_size = batch_size, epochs = n_epochs, callbacks=[checkpoint, early_stopping])
            
        logger.debug('Completed Pretraining.')
        if self.args.eval_after_pre_train:
            self.output.info('Starting Testing.')
            logger.debug('Starting Testing.')
            model.load_weights(f'{self.args.resDic}/{self.args.TF}_best_model.hdf5')

            loaded = np.load(f'{self.TFBSDic}/{self.TF}/test.npz')
            test_input1 = loaded['a']
            test_input1 = np.transpose(test_input1, [0,2,1])

            testScore = model.predict(test_input1, batch_size = batch_size)
            np.savetxt(fname = f'{self.args.resDic}/{self.args.TF}_testScore.txt', X = testScore,
                    delimiter = "\t")

            val_f1 = np.array(history_callback.history["val_f1"])
            train_f1 = np.array(history_callback.history["f1"])

            val_loss = np.array(history_callback.history["val_loss"])
            val_loss = val_loss[:,np.newaxis]
            train_loss = np.array(history_callback.history["loss"])
            train_loss = train_loss[:,np.newaxis]
            res_loss = np.concatenate((val_loss, train_loss), axis = 1)

            val_f1 = val_f1[:,np.newaxis]
            train_f1 = train_f1[:,np.newaxis]
            res_f1 = np.concatenate((val_f1, train_f1), axis = 1)

            np.savetxt(fname = f'{self.args.resDic}/{self.args.TF}_training_process.txt', X = res_f1,
                    delimiter = "\t", header = "\t".join(["val_f1","train_f1"]))
            np.savetxt(fname = f'{self.args.resDic}/{self.args.TF}_tracking_loss.txt', X = res_loss,
                    delimiter = "\t", header = "\t".join(["val_loss","train_loss"]))
            self.output.info('Completed Testing.')
            logger.debug('Completed Testing.')


    def process(self) -> None:
        self.output.info('Starting training data Process.')
        logger.debug('Starting training data Process.')

        self.deepTFBS_data_provider(self)
        self.pretraining(self)

        self.output.info('Completed training data Process.')
        logger.debug('Completed training data Process.')