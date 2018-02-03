import sys
import os
import numpy
#sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/Keras-0.3.1-py2.7.egg')
from keras.wrappers.scikit_learn import KerasClassifier
# Function to create model, required for KerasClassifierdef create_model(optimizer='adam'):
# Function to create model, required for KerasClassifierdef create_model():
from keras.models import Sequential, model_from_config, Model,load_model
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
from keras.callbacks import TensorBoard  
import copy
from evaluate.evalResult import evalResult

def predict(model,testing,testing_Y):
    predictions = model.predict(testing)
    acc, precision, sensitivity, specificity, MCC =evalResult.calculate_performace(predictions,testing_Y)
    return acc, precision, sensitivity, specificity, MCC    

def train_model(cnn,trainX_t,trainY,valX_t,valY,testX_t,testY,predict=False,weights=None,
                compiletimes=0,forkinas=False,transferlayer=1,earlystop=None,batch_size=1200,nb_epoch=500):
    print('bbbblllllllllllllll')
    if(predict is False):
        print(weights)
        if(weights is not None and compiletimes==0): #for the first time
            print('tlllllllllll')
            print("load weights:"+weights)
            if not forkinas:
                cnn.load_weights(weights);
            else:
                cnn2=copy.deepcopy(cnn)
                cnn2.load_weights(weights);
                for l in range((len(cnn2.layers)-transferlayer)): #the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                    #cnn.layers[l].trainable= False  # for frozen layer

        
        fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=2, validation_data=(valX_t, valY), callbacks=[earlystop])
    
    return cnn
       
        
    