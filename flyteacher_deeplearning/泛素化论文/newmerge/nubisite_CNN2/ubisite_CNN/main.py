import sys
import os
import numpy
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_config, Model,load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Input,MaxPooling1D,Conv1D
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
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import subprocess as sp
import scipy.stats as stats
import pickle
from keras import backend as K 
from keras.models import model_from_json 
from keras.callbacks import TensorBoard
from dataset.getData import get_data
from model.consModel import construct_cnn

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def create_model1(learn_rate=0.01, momentum=0,l2=0,num1=10,num2=10):
    model = Sequential()
    input_shape = (900,1)

    model.add(Conv1D(num1,kernel_size=3,
                         activation='relu',
                         kernel_regularizer=regularizers.l2(l2),
                         kernel_initializer=initializers.glorot_normal(seed=None),
                         input_shape=input_shape))  
    #model.add(Conv1D(100,kernel_size=3,activation='relu'))            
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(num2,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))
    
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model

def cv_param(training, y,valid_set_x,valid_y):
    
    model = KerasClassifier(build_fn=create_model,nb_epoch=50, batch_size=128, verbose=2)
    learn_rate = [0.001, 0.01, 0.1, 0.0001]
    momentum = [0.5, 0.6, 0.8, 0.9]
    l2 = [1.0, 0.5,0.05,0.005]
    num1 = [64,80,100,128]
    num2 = [64,80,100,128]
    param_grid = dict(learn_rate=learn_rate, momentum=momentum,l2=l2,num1=num1,num2=num2)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,n_iter = 20)
    grid_result = grid.fit(training, y)
    # summarize results 
    best_param = grid_result.best_estimator_.get_params()
    return best_param['learn_rate'],best_param['momentum'],best_param['l2'],best_param['num1'],best_param['num2']
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #for params, mean_score, scores in grid_result.grid_scores_:
    #    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
        
def run_DBN():
      
    # load data
    output = open('E:/research/PAPER/deeplearning/code/DeepDTIs/DeepDTIsDBNmaster/code/dataset1/ubisite300.pkl','rb')
    train_set,test_set= pickle.load(output,encoding='bytes')
    train_set_x1,train_set_y1 = train_set
    test_set_x,test_set_y = test_set
    output.close()
    
    r = np.random.permutation(len(train_set_y1))
    for i in range(len(r)):
        print(r[i])
    train_set_x = train_set_x1[r,:]
    train_set_y = train_set_y1[r]
    
    '''
    with open('r.txt','r') as fp:
        r = np.zeros((6620))
        j = -1
        for line in fp:
            j += 1
            r[j] = int(line)
    train_set_x = np.zeros((train_set_x1.shape[0],train_set_x1.shape[1]))
    train_set_y = np.zeros((train_set_x1.shape[0],1))
    for i in range(len(r)):
        print(int(r[i]))
        train_set_x[i,:] = train_set_x1[r[i],:]
        train_set_y[i,:] = train_set_y1[r[i]]
    '''

    # pre-deal
    #train_set_x = normal.class_train(train_set_x)
    #test_set_x = normal.class_test(test_set_x)
    train_set_x = normal.column_train(train_set_x)
    test_set_x = normal.column_test(test_set_x)
    #train_set_x = train_set_x.reshape((len(train_set_x), np.prod(train_set_x.shape[1:])))
    #test_set_x = test_set_x.reshape((len(test_set_x), np.prod(test_set_x.shape[1:])))
    
    plot.tsne(train_set_x,train_set_y)
    
    total_input = train_set_x.shape[1]
    y,encoder = preprocess_labels(train_set_y)
    #valid_y, encoder = preprocess_labels(valid_set_y, encoder = encoder)
    #cv_param(train_set_x,y)
    
    # construct training-predict evaluate
    ubisite_net = consModel.construct_model(train_set_x,num_hidden=total_input,sec_num_hidden=total_input,third_hidden=total_input,fourth_hidden=total_input,fifth_hidden=total_input)
    #plot.autoncoder(ubisite_net,train_set_x,train_set_y,num = 0)
    predictions,predict_y = trainModel.train_model(ubisite_net,total_input,train_set_x,y,train_set_y,test_set_x,test_set_y)
    for i in range(len(predictions)):
        print(predictions[i],' ',predict_y[i])
    evalResult.evaluate(predictions,predict_y,test_set_x,test_set_y)

def run_CNN():
    output = open('E:/research/PAPER/deeplearning/sites/ubisite2/dataset/win15.pkl','rb')
    train_set,test_set,valid_set= pickle.load(output,encoding='bytes')
    train_set_x,train_set_y = train_set
    test_set_x,test_set_y = test_set
    valid_set_x,valid_set_y = valid_set
    
    #train_set_x = normal.Z_ScoreNormalization(train_set_x,np.average(train_set_x),np.std(train_set_x))
    #test_set_x = normal.Z_ScoreNormalization(test_set_x,np.average(test_set_x),np.std(test_set_x))
    #valid_set_x = normal.Z_ScoreNormalization(valid_set_x,np.average(valid_set_x),np.std(valid_set_x))
    output.close()
    
    total_input = train_set_x.shape[1]
    # expand dims
    train_set_x = np.expand_dims(train_set_x,2)
    test_set_x = np.expand_dims(test_set_x,2)
    valid_set_x = np.expand_dims(valid_set_x,2)
    
    y,encoder = preprocess_labels(train_set_y)
    valid_y, encoder = preprocess_labels(valid_set_y, encoder = encoder)  
    
    #learn_rate,momentum,l2,num1,num2 = cv_param(train_set_x,y,valid_set_x,valid_y)
   
    # construct training-predict evaluate
    tencent_net = consModel.construct_cnn(train_set_x)
    predictions,predict_y = trainModel.train_model(tencent_net,total_input,train_set_x,y,train_set_y,test_set_x,test_set_y,valid_set_x,valid_y)
    '''
    for i in range(len(predictions)):
        
        if predictions[i]>0.6:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
        
        print(predictions[i],' ',predict_y[i])
    '''
    evalResult.evaluate(predictions,predict_y,test_set_x,test_set_y)    
    
def get_model():
    
    output = open('E:/research/tencentGame/test2/testAll_xyt.pkl','rb')
    test_set= pickle.load(output,encoding='bytes')
    test_set_x,test_set_y = test_set
    test_set_x = normal.Z_ScoreNormalization(test_set_x,np.average(test_set_x),np.std(test_set_x))
    
    test_set_x = np.expand_dims(test_set_x,2)
    ori_model = load_model('my_model_2219_22.h5')
    
    predictions,predict_y = trainModel.predict(ori_model,test_set_x)
    for i in range(len(predictions)):
        '''
        if predictions[i]>0.275:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
        '''       
        print(predictions[i],' ',predict_y[i])
    evalResult.evaluate(predictions,predict_y,test_set_x,test_set_y)     

if __name__ == "__main__":
    #run_DBN()
    get_data()
    #run_CNN()
    #get_model()
    