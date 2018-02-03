#Bootstrapping_allneg_continue
from multiCNN import MultiCNN
from DProcess import convertRawToXY
import pandas as pd
import numpy as np
import keras.models as models
from keras.models import Model


def bootStrapping_allneg_continue_val(trainfile = None,valfile=None,
                                      srate=1,nb_epoch1=3,nb_epoch2=30,earlystop=None,
                                      maxneg=None,codingMode=0,transferlayer=1,inputweights=None,
                                      outputweights=None,forkinas=False): #inputfile:fragments (n*34);srate:selection rate for positive data;nclass:number of class models
  
  trainX = trainfile
    # train_pos  train_neg
  train_pos=trainX[np.where(trainX[:,0]==1)]
  train_neg=trainX[np.where(trainX[:,0]!=1)]
  train_pos=pd.DataFrame(train_pos)
  train_neg=pd.DataFrame(train_neg) 

  a=int(train_pos.shape[0]*0.9)
  b=train_neg.shape[0]-int(train_pos.shape[0]*0.1)
  train_pos_ss=train_pos[0:a]
  train_neg_ss=train_neg[0:b]

  # val dataset
  val_pos=train_pos[(a+1):];
  val_neg=train_neg[(b+1):];
  val_all=pd.concat([val_pos,val_neg])
  valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode)
  val_physicalX,val_physicalY = convertRawToXY(val_all.as_matrix(),codingMode=3)
  '''
  c = int(train_pos_s.shape[0]*0.6);
  d = train_neg_s.shape[0]-int(train_pos_s.shape[0]*0.4)
  train_pos_ss = train_pos_s[0:c]
  train_neg_ss = train_neg_s[0:d]

  # test dataset
  test_pos = train_pos_s[(c+1):]
  test_neg = train_neg_s[(d+1):]
  test_all = pd.concat([test_pos,test_neg])
  testX1,testY1 = convertRawToXY(test_all.as_matrix(),codingMode=codingMode)
  '''
  slength=int(train_pos_ss.shape[0]*srate); 
  nclass=int(train_neg_ss.shape[0]/slength); 
  
  if(maxneg is not None):
    nclass=min(maxneg,nclass); #cannot do more than maxneg times  
  
  for I in range(nb_epoch1):
    train_neg_ss=train_neg_ss.sample(train_neg_ss.shape[0]); #shuffle neg sample
    train_pos_sss=train_pos_ss.sample(slength)
    for t in range(nclass):
        train_neg_sss=train_neg_ss[(slength*t):(slength*t+slength)];
        train_all=pd.concat([train_pos_sss,train_neg_sss])
        trainX1,trainY1 = convertRawToXY(train_all.as_matrix(),codingMode=codingMode)
        train_physicalX,train_physicalY = convertRawToXY(train_all.as_matrix(),codingMode=3)
        if t==0:
            models=MultiCNN(trainX1,trainY1,train_physicalX,train_physicalY,valX1,valY1,val_physicalX,val_physicalY,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t)
        else:
            models=MultiCNN(trainX1,trainY1,valX1,valY1,val_physicalX,val_physicalY,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t,compilemodels=models)
        
        print "modelweights assigned for "+str(t)+" bootstrap.\n";
        if(outputweights is not None):
            models.save_weights(outputweights,overwrite=True)
  
  
  return models;
