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
from plotfigure.plot import plot
from model.consModel import consModel
from model.trainModel import trainModel
from evaluate.evalResult import evalResult

class normal(object):

    def class_train(train_set_x):
        AAC_max = 0
        AAC_min = 100
        AAPC_max = 0
        AAPC_min = 100
        ASA_max = 0
        ASA_min = 100
        PSSM_max = 0 
        PSSM_min =100
        PWM_max = 0
        PWM_min = 100
        SS_max = 0
        SS_min =100
        # AAC
        for i in range(train_set_x.shape[0]):
            for j in range(1,21):
                if(train_set_x[i][j]>AAC_max):
                    AAC_max = train_set_x[i][j]
                if(train_set_x[i][j]<AAC_min):
                    AAC_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(1,21):
                train_set_x[i][j] =  (train_set_x[i][j] - AAC_min)/(AAC_max-AAC_min)   
        
        # AAPC   
        for i in range(train_set_x.shape[0]):
            for j in range(22,462):
                if(train_set_x[i][j]>AAPC_max):
                    AAPC_max = train_set_x[i][j]
                if(train_set_x[i][j]<AAPC_min):
                    AAPC_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(22,462):
                train_set_x[i][j] =  (train_set_x[i][j] - AAPC_min)/(AAPC_max-AAPC_min) 
        
        # ASA
        for i in range(train_set_x.shape[0]):
            for j in range(463,475):
                if(train_set_x[i][j]>ASA_max):
                    ASA_max = train_set_x[i][j]
                if(train_set_x[i][j]<ASA_min):
                    ASA_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(463,475):
                train_set_x[i][j] =  (train_set_x[i][j] - ASA_min)/(ASA_max-ASA_min)    
                
        # PSSM
        for i in range(train_set_x.shape[0]):
            for j in range(476,735):
                if(train_set_x[i][j]>PSSM_max):
                    PSSM_max = train_set_x[i][j]
                if(train_set_x[i][j]<PSSM_min):
                    PSSM_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(476,735):
                train_set_x[i][j] =  (train_set_x[i][j] - PSSM_min)/(PSSM_max-PSSM_min)     
        # PWM
        for i in range(train_set_x.shape[0]):
            for j in range(736,761):
                if(train_set_x[i][j]>PWM_max):
                    PWM_max = train_set_x[i][j]
                if(train_set_x[i][j]<PWM_min):
                    PWM_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(735,761):
                train_set_x[i][j] =  (train_set_x[i][j] - PWM_min)/(PWM_max-PWM_min)    
        # SS
        for i in range(train_set_x.shape[0]):
            for j in range(762,800):
                if(train_set_x[i][j]>SS_max):
                    SS_max = train_set_x[i][j]
                if(train_set_x[i][j]<SS_min):
                    SS_min = train_set_x[i][j]
        for i in range(train_set_x.shape[0]):
            for j in range(762,800):
                train_set_x[i][j] =  (train_set_x[i][j] - SS_min)/(SS_max-SS_min)
                
        return train_set_x
    
    def class_test(test_set_x):
        
        AAC_max = 0
        AAC_min = 100
        AAPC_max = 0
        AAPC_min = 100
        ASA_max = 0
        ASA_min = 100
        PSSM_max = 0 
        PSSM_min =100
        PWM_max = 0
        PWM_min = 100
        SS_max = 0
        SS_min =100
        # AAC
        for i in range(test_set_x.shape[0]):
            for j in range(1,21):
                if(test_set_x[i][j]>AAC_max):
                    AAC_max = test_set_x[i][j]
                if(test_set_x[i][j]<AAC_min):
                    AAC_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(1,21):
                test_set_x[i][j] =  (test_set_x[i][j] - AAC_min)/(AAC_max-AAC_min)   
        
        # AAPC   
        for i in range(test_set_x.shape[0]):
            for j in range(22,462):
                if(test_set_x[i][j]>AAPC_max):
                    AAPC_max = test_set_x[i][j]
                if(test_set_x[i][j]<AAPC_min):
                    AAPC_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(22,462):
                test_set_x[i][j] =  (test_set_x[i][j] - AAPC_min)/(AAPC_max-AAPC_min) 
        
        # ASA
        for i in range(test_set_x.shape[0]):
            for j in range(463,475):
                if(test_set_x[i][j]>ASA_max):
                    ASA_max = test_set_x[i][j]
                if(test_set_x[i][j]<ASA_min):
                    ASA_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(463,475):
                test_set_x[i][j] =  (test_set_x[i][j] - ASA_min)/(ASA_max-ASA_min)    
                
        # PSSM
        for i in range(test_set_x.shape[0]):
            for j in range(476,735):
                if(test_set_x[i][j]>PSSM_max):
                    PSSM_max = test_set_x[i][j]
                if(test_set_x[i][j]<PSSM_min):
                    PSSM_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(476,735):
                test_set_x[i][j] =  (test_set_x[i][j] - PSSM_min)/(PSSM_max-PSSM_min)     
        # PWM
        for i in range(test_set_x.shape[0]):
            for j in range(736,761):
                if(test_set_x[i][j]>PWM_max):
                    PWM_max = test_set_x[i][j]
                if(test_set_x[i][j]<PWM_min):
                    PWM_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(735,761):
                test_set_x[i][j] =  (test_set_x[i][j] - PWM_min)/(PWM_max-PWM_min)    
        # SS
        for i in range(test_set_x.shape[0]):
            for j in range(762,800):
                if(test_set_x[i][j]>SS_max):
                    SS_max = test_set_x[i][j]
                if(test_set_x[i][j]<SS_min):
                    SS_min = test_set_x[i][j]
        for i in range(test_set_x.shape[0]):
            for j in range(762,800):
                test_set_x[i][j] =  (test_set_x[i][j] - SS_min)/(SS_max-SS_min)    
        
        return test_set_x
    
    def column_train(train_set_x):
        for i in range(train_set_x.shape[1]):
            if(max(train_set_x[:,i])-min(train_set_x[:,i])!=0):
                train_set_x[:,i] = (train_set_x[:,i]-min(train_set_x[:,i]))/(max(train_set_x[:,i])-min(train_set_x[:,i]))
        
        return train_set_x
    
    def column_test(test_set_x):
        for i in range(test_set_x.shape[1]):
            if(max(test_set_x[:,i])-min(test_set_x[:,i])!=0):
                test_set_x[:,i] = (test_set_x[:,i]-min(test_set_x[:,i]))/(max(test_set_x[:,i])-min(test_set_x[:,i]))
         
        return test_set_x
    
    def normal_all(data_set):
        max_data = 0
        min_data = 100
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]):
                if(data_set[i][j]>max_data):
                    max_data = data_set[i][j]
                if(data_set[i][j]<min_data):
                    min_data = data_set[i][j]
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]):
                data_set[i][j] =  (data_set[i][j] - min_data)/(max_data-min_data)
        return data_set
    
    def Z_ScoreNormalization(data_set,mu,sigma):  
        data_set = (data_set - mu) / sigma;  
        return data_set;      
                