import sys
import os
import numpy
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-0.3.1-py2.7.egg')
import keras.layers.convolutional as conv
import keras.layers.core as core
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifierdef create_model(optimizer='adam'):
# Function to create model, required for KerasClassifierdef create_model():
from keras.models import Sequential, model_from_config, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input,MaxPooling1D,Conv1D,merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
#from keras.optimizers import kl_divergence
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import gzip
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import subprocess as sp
import scipy.stats as stats
import pickle
from keras import backend as K 
from keras.models import model_from_json  
from keras.regularizers import l1, l2
from model.attention import Attention,myFlatten
from model.trainModel import train_model

def construct_cnn(trainX, trainY,valX, valY,testX,testY,
         batch_size=1200, 
         nb_epoch=500,
         weights = None,
         earlystop=None,transferlayer=1,compiletimes=0,forkinas=False,
         compilemodels=None):
    
    input_row     = trainX.shape[2]
    input_col     = trainX.shape[3]
    
    trainX_t=trainX;
    valX_t=valX
    testX_t = testX
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    nb_epoch=100;#set to a very big value since earlystop used
    
    trainX_t.shape=(trainX_t.shape[0],input_row,input_col)
    valX_t.shape=(valX_t.shape[0],input_row,input_col)
    testX_t.shape = (testX_t.shape[0],input_row,input_col)
    
    if compiletimes==0: 
        
        input = Input(shape=(input_row,input_col))
        filtersize1= 2
        filtersize2= 10
        filtersize3= 15
        filter1=200
        filter2=150
        filter3=200
        dropout1=0.75
        dropout2=0.75
        dropout4=0.25
        dropout5=0.25
        dropout6=0
        L1CNN=0
        nb_classes=2
        batch_size=5000
        actfun="relu"; 
        optimization='adam';
        attentionhidden_x=10
        attentionhidden_xr=8
        attention_reg_x=0.151948
        attention_reg_xr=2
        dense_size1=149
        dense_size2=8
        dropout_dense1=0.298224
        dropout_dense2=0
        
        input = Input(shape=(input_row,input_col))
        x = conv.Convolution1D(filter1, filtersize1,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(input) 
        x = Dropout(dropout1)(x)
        x = Activation(actfun)(x)
        x = conv.Convolution1D(filter2,filtersize2,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
        x = Dropout(dropout2)(x)
        x = Activation(actfun)(x)
        #x = conv.Convolution1D(filter3,filtersize3,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
        #x = Activation(actfun)(x)
        x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
        
        x = Dropout(dropout4)(x)
        x_reshape=Dropout(dropout5)(x_reshape)
        
        decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
        decoded_x=decoder_x(x)
        output_x = myFlatten(x._keras_shape[2])(decoded_x)
        
        decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
        decoded_xr=decoder_xr(x_reshape)
        output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
        
        output=merge([output_x,output_xr],mode='concat')
        output=Dropout(dropout6)(output)
        output=Dense(dense_size1,init='he_normal',activation='relu')(output)
        output=Dropout(dropout_dense1)(output)
        output=Dense(dense_size2,activation="relu",init='he_normal')(output)
        output=Dropout(dropout_dense2)(output)
        out=Dense(nb_classes,init='he_normal',activation='softmax')(output)
        cnn=Model(input,out)
        cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
         
    else:
        cnn=compilemodels
    
    model = train_model(cnn,trainX_t,trainY,valX_t,valY,testX_t,testY,predict=False,weights=None,compiletimes=compiletimes,forkinas=forkinas,transferlayer=transferlayer,earlystop = early_stopping,batch_size=batch_size,nb_epoch=nb_epoch)
    
    return model,testX_t        