# -*- coding: utf-8 -*-
"""
# @Time    : 2018/6/10 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import os
# import theano
import tensorflow
import time
import numpy as np
import pandas as pd
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import merge
from keras.layers import pooling
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input,normalization
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from tensorflow.python.util import nest
# from theano.util import nest
#导入相关层的结构
#from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np


from keras.optimizers import Nadam, Adam, SGD
import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .DTP2_evaluate_cls import DTP_evaluate_modle

class DTP_set_net(object):
    
    def zyh_CNN(self,trainX, trainY, valX=None, valY=None, compiletimes=0, forkinas=False, transferlayer=1, compilemodels=None,
                earlystop=None, nb_epoch=500, batch_size=7124, fildername="fildername"):
        """
        :argument
        :return:
        """
        input_row = trainX.shape[1]
        input_col = trainX.shape[2]
        print(trainX.shape)

        trainX_t = trainX;
        valX_t = valX;

        trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
        if (valX is not None):
            valX_t.shape = (valX_t.shape[0], input_row, input_col)
        # print(valX_t)
        if (earlystop is not None):
            early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
            nb_epoch = 1000;  # set to a very big value since earlystop used
    
        if compiletimes == 0:
            input = Input(shape=(input_row, input_col))
            filter1 = 64
            filtersize1 = 2
            strides = 1
            dropout1 = 0.25
            L1CNN = 0
            nb_classes = 2
            batch_size = batch_size
            actfun = "relu";
            nadam = Nadam(lr=0.00001)
            optimization = nadam
    
            dense_size1 = 128
            dense_size2 = 64
            dense_size3 = 8
            dropout_dense1 = 0.298224
            dropout_dense2 = 0
            dropout_dense3 = 0
            input = Input(shape=(input_row, input_col))
            x = conv.Conv1D(filter1, filtersize1,strides =strides , init='glorot_normal', W_regularizer=regularizers.l2(L1CNN),
                            border_mode="same")(input)
            x = Dropout(dropout1)(x)
            x = Activation(actfun)(x)
            x = core.Flatten()(x)
    
            output = x
            output = BatchNormalization()(output)
            output = Dropout(dropout1)(output)
            output = Dense(dense_size1, init='glorot_normal', activation='relu', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense1)(output)
            output = Dense(dense_size2, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense2)(output)
            output = Dense(dense_size3, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense3)(output)
    
            output = BatchNormalization()(output)
            out = Dense(nb_classes, init='glorot_normal', activation='softmax')(output)
            cnn = Model(input, out)
            cnn.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
            pass
        else:
            print("use old net")
            cnn = compilemodels
            pass
        """
        if (weights is not None and compiletimes == 0):  # for the first time
            # print "load weights:"+weights;
            # print ("load weights:" + weights)
            if not forkinas:
                cnn.load_weights(weights);
            else:
                cnn2 = copy.deepcopy(cnn)
                cnn2.load_weights(weights);
                for l in range((len(cnn2.layers) - transferlayer)):  # the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())"""
        if (valX is not None):
            if (earlystop is None):
                print("!")
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(valX_t, valY))
            else:
                print("@")
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(valX_t, valY), callbacks=[checkpointer, early_stopping])
        else:
            print("#")
            fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch)
            print(fitHistory)
    
        # fitHistory = fitHistory
        eva_modle = DTP_evaluate_modle()
    
        # print_loss(fitHistory, compiletimes, 'loss_','result_1')
        # print_acc(fitHistory, compiletimes, 'acc_','result_1')
        train_loss1, train_acc1, val_loss2, val_acc2 = eva_modle.print_loss_acc(fitHistory, compiletimes, 'loss_acc_', fildername)
        accuracys = {'train_loss1': train_loss1, 'train_acc1': train_acc1, 'val_loss2': val_loss2, 'val_acc2': val_acc2}
        # print(accuracys)
    
        return cnn, accuracys
    
    def zyh_CNN_one_hot_flying(self,trainX, trainY, valX=None, valY=None, compiletimes=0, forkinas=False, transferlayer=1, compilemodels=None,
                earlystop=None, nb_epoch=500, batch_size=4096, fildername=None):
        """
        :argument
        :return:
        """

        input_row = trainX.shape[1]
        input_col = trainX.shape[2]

        trainX_t = trainX;
        valX_t = valX;

    
        trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
        if (valX is not None):
            valX_t.shape = (valX_t.shape[0], input_row, input_col)
        # print(valX_t)
        if (earlystop is not None):
            early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
            nb_epoch = 1000;  # set to a very big value since earlystop used
    
        if compiletimes == 0:
            input = Input(shape=(input_row, input_col))
            filter1 = 64
            filtersize1 = 2
            dropout1 = 0.25
            L1CNN = 0
            nb_classes = 2
            batch_size = batch_size
            actfun = "relu";
            nadam = Nadam(lr=0.00001)
            optimization = nadam
    
            dense_size1 = 128
            dense_size2 = 64
            dense_size3 = 8
            dropout_dense1 = 0.298224
            dropout_dense2 = 0
            dropout_dense3 = 0
            input = Input(shape=(input_row, input_col))
            """
            x = conv.Conv1D(filter1, filtersize1, init='glorot_normal', W_regularizer=regularizers.l2(L1CNN),
                            border_mode="same")(input)
            x = Dropout(dropout1)(x)
            x = Activation(actfun)(x)
            x = core.Flatten()(x)
    
            output = x
            output = BatchNormalization()(output)
            output = Dropout(dropout1)(output)
            output = Dense(dense_size1, init='glorot_normal', activation='relu', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense1)(output)
            output = Dense(dense_size2, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense2)(output)
            output = Dense(dense_size3, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense3)(output)
    
            output = BatchNormalization()(output)
            out = Dense(nb_classes, init='glorot_normal', activation='softmax')(output)
    """

            strides = 1
            kernal_size_times = 20
            ########## Begin Oneofkey Network ##########
            x = conv.Convolution1D(201, 2*kernal_size_times, strides = strides ,init='glorot_normal', W_regularizer=regularizers.l1(L1CNN), border_mode="same", name='0')(input)
            x = Dropout(0.4)(x)
            x = Activation('softsign')(x)
    
            x = conv.Convolution1D(151, 3*kernal_size_times, strides = strides,init='glorot_normal', W_regularizer=regularizers.l2(L1CNN), border_mode="same", name='1')(x)
            x = Dropout(0.4)(x)
            x = Activation('softsign')(x)
    
            x = conv.Convolution1D(151, 5*kernal_size_times,strides = strides , init='glorot_normal', W_regularizer=regularizers.l2(L1CNN), border_mode="same", name='2')(x)
            x = Dropout(0.4)(x)
            x = Activation('softsign')(x)
    
            x = conv.Convolution1D(101, 7*kernal_size_times, strides = strides ,init='glorot_normal', W_regularizer=regularizers.l2(L1CNN), border_mode="same", name='3')(x)
            x = Activation('softsign')(x)
            x_reshape = core.Reshape((x._keras_shape[2], x._keras_shape[1]))(x)
            x = Dropout(0.4)(x)
    
            output_x = core.Flatten()(x)
            output = BatchNormalization()(output_x)
            output = Dropout(0)(output)
    
            output = Dense(256, init='glorot_normal', activation='relu', name='4')(output)
            output = Dropout(0.298224)(output)
            output = Dense(128, init='glorot_normal', activation="relu", name='5')(output)
            output = Dropout(0)(output)
            output = Dense(128, activation="relu", init='glorot_normal', name='6')(output)
            output = Dropout(0)(output)
            out = Dense(nb_classes, init='glorot_normal', activation='softmax')(output)
            ########## End Oneofkey Network ##########
    
    
            cnn = Model(input, out)
            cnn.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
            pass
        else:
            print("use old net")
            cnn = compilemodels
            pass
        """
        if (weights is not None and compiletimes == 0):  # for the first time
            # print "load weights:"+weights;
            # print ("load weights:" + weights)
            if not forkinas:
                cnn.load_weights(weights);
            else:
                cnn2 = copy.deepcopy(cnn)
                cnn2.load_weights(weights);
                for l in range((len(cnn2.layers) - transferlayer)):  # the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())"""
        if (valX is not None):
            if (earlystop is None):
                print("!")
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(valX_t, valY))
            else:
                print("@")
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(valX_t, valY), callbacks=[checkpointer, early_stopping])
        else:
            print("#")
            fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch)
            print(fitHistory)
    
        # fitHistory = fitHistory
    
        # print_loss(fitHistory, compiletimes, 'loss_','result_1')
        # print_acc(fitHistory, compiletimes, 'acc_','result_1')
        eva_modle = DTP_evaluate_modle()
        train_loss1, train_acc1, val_loss2, val_acc2 = eva_modle.print_loss_acc(fitHistory, compiletimes, 'loss_acc_', fildername)
    
        accuracys = {'train_loss1': train_loss1, 'train_acc1': train_acc1, 'val_loss2': val_loss2, 'val_acc2': val_acc2}
        # print(accuracys)
    
        return cnn, accuracys
    
    def zyh_CNN_mnist(self,trainX, trainY, valX=None, valY=None, compiletimes=0, forkinas=False, transferlayer=1, compilemodels=None,
                earlystop=None, nb_epoch=500, batch_size=1024, fildername="fildername"):
        """
        :argument
        :return:
        """
        X_train = trainX;
        X_test = valX;
        
        #input image size 28*28
        img_rows , img_cols = trainX.shape[1], trainX.shape[2]
        print(img_rows , img_cols)
        
        #reshaping
        #"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        #more reshaping
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)        
        

        if (earlystop is not None):
            early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
            nb_epoch = 1000;  # set to a very big value since earlystop used

        if compiletimes == 0:
            ## model building
            model = Sequential()
            #convolutional layer with rectified linear unit activation
            model.add(Conv2D(32, kernel_size=(1, 3),
                             activation='relu',
                             input_shape=input_shape))
            #32 convolution filters used each of size 3x3
            #again
            model.add(Conv2D(64, (1, 5), activation='relu'))
            #64 convolution filters used each of size 3x3
            #choose the best features via pooling
            model.add(MaxPooling2D(pool_size=(2, 2)))
            #randomly turn neurons on and off to improve convergence
            model.add(Dropout(0.25))
            #flatten since too many dimensions, we only want a classification output
            model.add(Flatten())
            #fully connected to get all relevant data
            model.add(Dense(128, activation='relu'))
            #one more dropout for convergence' sake :) 
            model.add(Dropout(0.5))
            #output a softmax to squash the matrix into output probabilities
            model.add(Dense(2, activation='softmax'))            
            #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
            #categorical ce since we have multiple classes (10) 
            """model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy']) """      
            model.compile(loss='binary_crossentropy',
                                  optimizer=Nadam(lr=0.00001),
                                  metrics=['accuracy'])              
            """
            input = Input(shape=(input_row, input_col))
            filter1 = 128
            filtersize1 = 93
            strides = 15
            dropout1 = 0.25
            L1CNN = 0
            nb_classes = 2
            batch_size = batch_size
            actfun = "relu";
            nadam = Nadam(lr=0.00001)
            optimization = nadam

            dense_size1 = 128
            dense_size2 = 64
            dense_size3 = 8
            dropout_dense1 = 0.298224
            dropout_dense2 = 0
            dropout_dense3 = 0
            input = Input(shape=(input_row, input_col))
            x = conv.Conv1D(filter1, filtersize1,strides =strides , init='glorot_normal', W_regularizer=regularizers.l2(L1CNN),
                            border_mode="same")(input)
            x = Dropout(dropout1)(x)
            x = Activation(actfun)(x)
            x = core.Flatten()(x)

            output = x
            output = BatchNormalization()(output)
            output = Dropout(dropout1)(output)
            output = Dense(dense_size1, init='glorot_normal', activation='relu', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense1)(output)
            output = Dense(dense_size2, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense2)(output)
            output = Dense(dense_size3, activation="relu", init='glorot_normal', W_regularizer=regularizers.l2(L1CNN))(output)
            output = Dropout(dropout_dense3)(output)

            output = BatchNormalization()(output)
            out = Dense(nb_classes, init='glorot_normal', activation='softmax')(output)
            cnn = Model(input, out)
            cnn.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
            """
            
            
            
            
            pass
        else:
            print("use old net")
            model = compilemodels
            pass
 
        if (valX is not None):
            if (earlystop is None):
                print("!")
                fitHistory = model.fit(X_train, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(X_test, valY))
            else:
                print("@")
                fitHistory = model.fit(X_train, trainY, batch_size=batch_size, epochs=nb_epoch,
                                     validation_data=(valX_t, valY), callbacks=[checkpointer, early_stopping])
        else:
            print("#")
            fitHistory = model.fit(X_train, trainY, batch_size=batch_size, epochs=nb_epoch)
            print(fitHistory)

        # fitHistory = fitHistory
        eva_modle = DTP_evaluate_modle()

        # print_loss(fitHistory, compiletimes, 'loss_','result_1')
        # print_acc(fitHistory, compiletimes, 'acc_','result_1')
        train_loss1, train_acc1, val_loss2, val_acc2 = eva_modle.print_loss_acc(fitHistory, compiletimes, 'loss_acc_', fildername)
        accuracys = {'train_loss1': train_loss1, 'train_acc1': train_acc1, 'val_loss2': val_loss2, 'val_acc2': val_acc2}
        # print(accuracys)

        return model, accuracys
    