import sys
import os
import numpy
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-0.3.1-py2.7.egg')
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifierdef create_model(optimizer='adam'):
# Function to create model, required for KerasClassifierdef create_model():
from keras.models import Sequential, model_from_config, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
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

class plot(object):
    def create_Autoencoder(data_dim):
        encoding_dim = 2
        input_data = Input(shape=(data_dim,))
        
        # encode
        encoded = Dense(128, activation='relu')(input_data)  
        encoded = Dense(64, activation='relu')(encoded)  
        encoded = Dense(10, activation='relu')(encoded)  
        encoder_output = Dense(encoding_dim)(encoded) 
        
        # decode
        decoded = Dense(10, activation='relu')(encoder_output)  
        decoded = Dense(64, activation='relu')(decoded)  
        decoded = Dense(128, activation='relu')(decoded)  
        decoded = Dense(data_dim, activation='tanh')(decoded) 
        
        autoencoder = Model(inputs=input_data, outputs=decoded)  
        encoder = Model(inputs=input_data, outputs=encoder_output)
        
        return autoencoder,encoder       
    def tsne(data_set,data_y):
        X_tsne = TSNE(n_components=2).fit_transform(data_set)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_y,marker='o')
        plt.show()
    
    def tsne_hidden(origin_model,data_set,data_y,num):
        get_layer_out = K.function([origin_model.layers[0].input,K.learning_phase()],origin_model.layers[num].output) 
        layer_out = get_layer_out([data_set,1])
        print(layer_out.shape) 
        X_tsne = TSNE(n_components=2).fit_transform(layer_out)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_y,marker='o')
        plt.show()        
        
    def autoncoder(origin_model,data_set,data_y,num):
        if(num==0):
            # autoencoder
            print(data_set.shape[1])
            autoencoder,encoder = plot.create_Autoencoder(data_set.shape[1])
        
            # compile autoencoder
            #autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 
            
            # training
            #autoencoder.fit(data_set, data_set,batch_size=25, nb_epoch=20,shuffle=True)    
            
            # plot
            data_set = encoder.predict(data_set) 
            plt.scatter(data_set[:, 0], data_set[:, 1], c=data_y, s=15)  
            plt.colorbar()  
            plt.show()         
        else:
            get_layer_out = K.function([origin_model.layers[0].input,K.learning_phase()],origin_model.layers[num].output) 
            layer_out = get_layer_out([data_set,1])
            print(layer_out.shape)  
            
            # autoencoder
            print(layer_out.shape[1])
            autoencoder,encoder = plot.create_Autoencoder(layer_out.shape[1])
            
            # compile autoencoder
            #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')        
            
            # training
            #autoencoder.fit(layer_out, layer_out,batch_size=25, nb_epoch=20,shuffle=True)    
            
            # plot
            layer_out = encoder.predict(layer_out) 
            plt.scatter(layer_out[:, 0], layer_out[:, 1], c=data_y, s=15)  
            plt.colorbar()  
            plt.show() 
     
        