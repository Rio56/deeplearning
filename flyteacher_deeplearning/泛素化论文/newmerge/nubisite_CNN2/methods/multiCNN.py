import os
import theano
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
from keras.engine.topology import Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import WeightRegularizer, l1, l2
from attention import Attention,myFlatten,DenseAttention
from keras import backend as K
import copy

def MultiCNN(trainX, trainY, trainPhysicalX=None,trainPhysicalY=None,
             valX=None, valY=None, valPhysicalX=None, valPhysicalY = None,
             batch_size=1200, 
             nb_epoch=500,
             earlystop=None,transferlayer=1,weights=None,forkinas=False,compiletimes=0,
             compilemodels=None,predict=False):
    input_row  = trainX.shape[2]
    input_col  = trainX.shape[3]
    
    physical_row = trainPhysicalX.shape[2]
    physical_col = trainPhysicalX.shape[3]
    
    trainX_t=trainX;
    valX_t=valX;
    
    train_physical_X_t = trainPhysicalX
    val_physical_x_t = valPhysicalX
    
    
    if(earlystop is not None): 
        early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
        nb_epoch=500;#set to a very big value since earlystop used
    
    trainX_t.shape=(trainX_t.shape[0],input_row,input_col)
    if(valX is not None):
        valX_t.shape=(valX_t.shape[0],input_row,input_col)
    
    train_physical_X_t.shape = (train_physical_X_t.shape[0], physical_row,physical_col)
    val_physical_x_t.shape  = (val_physical_x_t.shape[0],physical_row,physical_col)
        
    input = Input(shape=(input_row,input_col))
    physicalInput = Input(shape=(physical_row,physical_col))
    
    if compiletimes==0:         
        #input = Input(shape=(input_row,input_col))
        filtersize1=2
        filtersize2=3
        filtersize3=9
       
        filter1=200
        filter2=150
        filter3=200
        filter4 = 150
        dropout1=0.25
        dropout2=0.25
        dropout4=0.25
        dropout5=0.25
        dropout6=0
        L1CNN=0
        nb_classes=2
        batch_size=15
        actfun="relu"; 
        optimizer = SGD(lr=0.001, momentum=0.9)
        optimization='Nadam';
        attentionhidden_x=10
        attentionhidden_xr=8
        attention_reg_x=0.151948
        attention_reg_xr=2
        dense_size1=149
        dense_size2=50
        dense_size3=100
        dropout_dense1=0.298224
        dropout_dense2=0
       
        x = conv.Convolution1D(filter1, filtersize1,init='glorot_normal',W_regularizer= l1(L1CNN),border_mode="same")(input) 
        x = Dropout(dropout1)(x)
        x = Activation(actfun)(x)
        x = conv.Convolution1D(filter2,filtersize2,init='glorot_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
        x = Dropout(dropout2)(x)
        x = Activation(actfun)(x)
        x = conv.Convolution1D(filter3,filtersize3,init='glorot_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
        x = Activation(actfun)(x)                   
        x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
         
        x = Dropout(dropout4)(x)
        x_reshape=Dropout(dropout5)(x_reshape)
        
        decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='glorot_normal',W_regularizer=l1(attention_reg_x)) # success  
        decoded_x=decoder_x(x)
        output_x = myFlatten(x._keras_shape[2])(decoded_x)
        
        decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='glorot_normal',W_regularizer=l1(attention_reg_xr))
        decoded_xr=decoder_xr(x_reshape)
        output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
        '''
         if nmodeltest == 1:
             output = output_x 
         elif nmodeltest == 2:
             output = output_xr
         elif nmodeltest == 3:
             output = myFlatten(x._keras_shape[2])(x)
         else:
         '''
        
        #begain sub-network
        physical_code_x = core.Flatten()(physicalInput)
        physical_code_x = Dense(2000,init='glorot_normal',activation='relu')(physical_code_x) 
        physical_code_x = BatchNormalization()(physical_code_x)        
        physical_code_x = Dense(1000,init='glorot_normal',activation='relu')(physical_code_x) 
        physical_code_x = BatchNormalization()(physical_code_x)
        physical_code_x = Dropout(0.75)(physical_code_x)
        physical_code_x = Dense(500,activation="relu",init='glorot_normal')(physical_code_x)
        physical_code_x = BatchNormalization()(physical_code_x)
        physical_code_x = Dropout(0.5)(physical_code_x)
        output_physical_x = Dense(100,activation="relu",init='glorot_normal')(physical_code_x)
        #output_physical_x_reshape=core.Reshape((output_physical_x._keras_shape[2],output_physical_x._keras_shape[1]))(output_physical_x)
        #decoder_physical_x = DenseAttention(hidden=50,activation='linear',init='uniform',W_regularizer=l1(attention_reg_x)) # success  
        #decoded_physical_x = decoder_physical_x(physical_code_x)
            
        #output_physical_x = myFlatten(decoded_physical_x._keras_shape[1])(decoded_physical_x)
        #output_physical_x = merge([physical_code_x,decoded_physical_x],mode="concat", concat_axis= 0)

        #end sub-network
        
        output = merge([output_x,output_xr],mode='concat')
     
            
            
        output=Dropout(dropout6)(output)
        output=Dense(dense_size1,init='glorot_normal',activation='relu')(output)
        output=Dropout(dropout_dense1)(output)
        output=Dense(dense_size2,activation="relu",init='glorot_normal')(output)
        output=Dropout(dropout_dense2)(output)
        
        output = merge([output,output_physical_x],mode='concat')
        output = BatchNormalization()(output)
        output=Dense(dense_size3,activation="relu",init='glorot_normal')(output)
        output = BatchNormalization()(output)
        out=Dense(nb_classes,init='glorot_normal',activation='softmax')(output)
        #out=Dense(nb_classes,init='glorot_normal',activation='softmax')(output)
        cnn=Model([input,physicalInput],out)
        cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
        
    else:
        cnn=compilemodels
    
    if(predict is False):
        if(weights is not None and compiletimes==0): #for the first time
            print "load weights:"+weights;
            if not forkinas:
                cnn.load_weights(weights);
            else:
                cnn2=copy.deepcopy(cnn)
                cnn2.load_weights(weights);
                for l in range((len(cnn2.layers)-transferlayer)): #the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                   #cnn.layers[l].trainable= False  # for frozen layer
        
        if(valX is not None):
            if(earlystop is None):
                #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(valX_t, valY))
                fitHistory = cnn.fit([trainX_t,train_physical_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=([valX_t,val_physical_x_t], valY))
                
            else:
                #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(valX_t, valY), callbacks=[early_stopping])
                fitHistory = cnn.fit([trainX_t,train_physical_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([valX_t,val_physical_x_t], valY), callbacks=[early_stopping])
        else:
            fitHistory = cnn.fit([trainX_t,train_physical_X_t], trainY, batch_size=batch_size, nb_epoch=nb_epoch)
            #fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
    
    return cnn