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

class evalResult(object):
    
    def calculate_performace(predictions,testing_Y):
        
        tp =0
        fp = 0
        tn = 0
        fn = 0
        predict_y = np.zeros(len(predictions))
        for i in range(len(predictions)):
            if (predictions[i][0]>0.5):
                predict_y[i] = 1
            else:
                predict_y[i] = 0
        for index in range(len(predictions)):
            if testing_Y[index][0] ==1:
                if testing_Y[index][0] == predict_y[index]:
                    tp = tp +1
                else:
                    fn = fn + 1   
            else:
                if testing_Y[index][0] == predict_y[index]:
                    tn = tn +1
                else:
                    fp = fp + 1
       
        print(tp,' ',fn,' ',tn,' ',fp)
        
        acc = float(tp + tn)/test_num
        precision = float(tp)/(tp+ fp)
        sensitivity = float(tp)/ (tp+fn)
        specificity = float(tn)/(tn + fp)
        MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        return acc, precision, sensitivity, specificity, MCC    
    
    def evaluate(predictions,predict_y,data_set,data_y):
        print(predictions.shape)
        #auc = roc_auc_score(data_set,predictions)
        test_num = len(data_set)
        acc, precision, sensitivity, specificity, MCC = evalResult.calculate_performace(test_num,predict_y,data_y)
        #print ("Test AUC: ", auc)
        print("TEST ACC",acc)
        print("TEST precision",precision)
        print("TEST sensitivity",sensitivity)
        print('TEST specificity',specificity)
        print('TEST MCC',MCC)    