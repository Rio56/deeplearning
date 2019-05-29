# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import os
#import theano
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
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from tensorflow.python.util import nest
#from theano.util import nest

from keras.optimizers import Nadam, Adam, SGD
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_loss_acc(fitHistory, i, type,fildername):
	#  ######### Print Loss Map ##########
	plt.figure()
	plt.plot(fitHistory.history['loss'])
	plt.plot(fitHistory.history['val_loss'])
	plt.plot(fitHistory.history['acc'])
	plt.plot(fitHistory.history['val_acc'])	
	# plt.title('size:%d' % size)
	plt.title('LOSS:times %d' % i)
	plt.ylim([0, 1.0])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.grid(True)
	plt.legend(['train_loss', 'validation_loss','train_acc', 'validation_acc'], loc='upper left')
	plt.savefig('./'+fildername+'/' + str(type) + '%d.png' % i)

	#  ############### final ################
	loss1 = fitHistory.history['loss'][-1]
	acc1 = fitHistory.history['acc'][-1]
	loss2 = fitHistory.history['val_loss'][-1]
	acc2 = fitHistory.history['val_acc'][-1]

	return loss1, acc1, loss2, acc2

def print_acc(fitHistory, i, type, fildername):
	#  ######### Print Loss Map ##########
	plt.figure()
	plt.plot(fitHistory.history['acc'])
	plt.plot(fitHistory.history['val_acc'])
	# plt.title('size:%d' % size)
	plt.title('ACC:times %d' % i)
	plt.ylim([0, 1.0])
	plt.ylabel('acc')
	plt.xlabel('epoch')
	plt.grid(True)
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('./'+fildername+'/' + str(type) + '%d.png' % i)

	#  ############### final ################
	loss1 = fitHistory.history['loss'][0]
	acc1 = fitHistory.history['acc'][0]
	loss2 = fitHistory.history['val_loss'][0]
	acc2 = fitHistory.history['val_acc'][0]

	return loss1, acc1, loss2, acc2

def print_loss(fitHistory, i, type, fildername):
	#  ######### Print Loss Map ##########
	plt.figure()
	plt.plot(fitHistory.history['loss'])
	plt.plot(fitHistory.history['val_loss'])
	# plt.title('size:%d' % size)
	plt.title('LOSS:times %d' % i)
	plt.ylim([0, 1.0])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.grid(True)
	plt.legend(['train', 'validation'], loc='upper left')
	plt.savefig('./'+fildername+'/' + str(type) + '%d.png' % i)

	#  ############### final ################
	loss1 = fitHistory.history['loss'][0]
	acc1 = fitHistory.history['acc'][0]
	loss2 = fitHistory.history['val_loss'][0]
	acc2 = fitHistory.history['val_acc'][0]

	return loss1, acc1, loss2, acc2

def zyh_CNN(trainX, trainY,valX=None, valY=None, compiletimes = 0, forkinas=False,transferlayer=1,compilemodels=None,earlystop=None,nb_epoch=750,batch_size = 4096,fildername = None ):
	"""
	:argument
	:return:
	"""
	input_row = trainX.shape[2]
	input_col = trainX.shape[3]
	trainX_t = trainX;
	valX_t = valX;
	#print(input_row)
	#print(input_col)
	
	trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
	if (valX is not None):
		valX_t.shape = (valX_t.shape[0], input_row, input_col)
	#print(valX_t)
	if (earlystop is not None):
		early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
		nb_epoch = 1000;  # set to a very big value since earlystop used
		
	if compiletimes == 0:
		input = Input(shape=(input_row, input_col))
		filter1 = 64
		filtersize1 = 4
		dropout1 = 0.25
		L1CNN = 0
		nb_classes = 2
		batch_size = batch_size
		actfun = "relu";
		nadam = Nadam(lr=0.00001)
		optimization = nadam
		
		dense_size1 = 256
		dense_size2 = 128
		dense_size3 = 8
		dropout_dense1 = 0.298224
		dropout_dense2 = 0.25
		dropout_dense3 = 0
		input = Input(shape=(input_row, input_col))
		x = conv.Conv1D(filter1, filtersize1, init='glorot_normal', W_regularizer=regularizers.l2(L1CNN),
		                       border_mode="same")(input)
		x = Dropout(dropout1)(x)
		x = Activation(actfun)(x)
		x = core.Flatten()(x)
		output = x
		output = Dropout(dropout1)(output)
		output = Dense(dense_size1, init='glorot_normal', activation='relu')(output)
		output = Dropout(dropout_dense1)(output)
		output = Dense(dense_size2, activation="relu", init='glorot_normal')(output)
		output = Dropout(dropout_dense2)(output)
		output = Dense(dense_size3, activation="relu", init='glorot_normal')(output)
		output = Dropout(dropout_dense3)(output)		
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
			print ("!")
			#print(valX_t.shape)
			#print(valY)
			fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
			                     validation_data=(valX_t, valY))
		else:
			print ("@")
			fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch,
			                     validation_data=(valX_t, valY), callbacks=[checkpointer,early_stopping])
	else:
		print ("#")
		fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, epochs=nb_epoch)
		print(fitHistory)
		
	#fitHistory = fitHistory
	
	#print_loss(fitHistory, compiletimes, 'loss_','result_1')
	#print_acc(fitHistory, compiletimes, 'acc_','result_1')
	train_loss1, train_acc1, val_loss2, val_acc2 = print_loss_acc(fitHistory, compiletimes, 'loss_acc_',fildername)
	
	
	accuracys = {'train_loss1':train_loss1, 'train_acc1':train_acc1, 'val_loss2':val_loss2, 'val_acc2':val_acc2}
	#print(accuracys)	
	
	      
	
	return cnn,accuracys


