import pandas as pd
import numpy as np
import keras.models as models
from keras.models import Model
from dataset.DProcess import convertRawToXY
from model.consModel import construct_cnn
from model.trainModel import train_model,predict

def find(i,id_num):
    for j in range(len(id_num)):
        if (id_num[j]==i):
            return True
    return False

def subSeqq(seq,id,num):
    win = num
    subSeq = ''
    if (id-win-1)<0 and (id + win+1)>len(seq):
        for i in range(win-id+1):
            subSeq+='X'
        for i in range(0,len(seq)-id-1):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='X'
    elif (id-win-1)<0 and (id+win+1)<=len(seq):
        for i in range(win-id+1):
            subSeq+='X'
        for i in range(0,id+win+1-1):
            subSeq+=seq[i]
    elif (id-win-1)>=0 and (id+win+1) > len(seq):
        for i in range(id-win-1,len(seq)-1):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='X'
    elif (id-win-1)>=0 and (id+win+1) <= len(seq):
        for i in range(id-win-1,id+win+1-1):
            subSeq+=seq[i]
    return subSeq    

def read_fasta(fasta_file, fragment_length):
    seq_list= []
    num = 0
    id_num=[]
    with open(fasta_file,'r') as fp:
        neg = 0
        pos = 0
        for line in fp:
            num += 1
            if (num == 2):
                line1 = line.split('\t')
                name1 = line1[1]
                id = int(line1[2])
                seq1 = line1[4]
                id_num.append(id-1)
            elif (num > 2):
                line1 = line.split('\t')
                name = line1[1]
                seq = line1[4]
                if (name == name1):
                    id_num.append(int(line1[2])-1)
                else:
                    for i in range(len(seq1)):
                        if (seq1[i]=='K' and find(i,id_num)):
                            pos += 1
                            subSeq = subSeqq(seq1,i,fragment_length)
                            final_seq = [1] + [AA for AA in subSeq]
                            seq_list.append(final_seq)
                        elif (seq1[i]=='K' and not find(i,id_num)):
                            neg += 1
                            subSeq = subSeqq(seq1,i,fragment_length)
                            final_seq = [0] + [AA for AA in subSeq]  
                            seq_list.append(final_seq)
                    id_num = []
                    name1 =  name
                    seq1 = seq
                    id_num.append(int(line1[2])-1)
        for i in range(len(seq1)):
            if (seq1[i]=='K' and find(i,id_num)):
                pos += 1
                subSeq = subSeqq(seq1,i,fragment_length)
                final_seq = [1] + [AA for AA in subSeq]
                seq_list.append(final_seq)
            elif (seq1[i]=='K' and not find(i,id_num)):
                neg += 1
                subSeq = subSeqq(seq1,i,fragment_length)
                final_seq = [0] + [AA for AA in subSeq]  
                seq_list.append(final_seq)  
        print(pos,' ',neg)
        return seq_list
                            
        
def get_data(srate = 0.8,codingMode=0,nb_epoch1=1,nb_epoch2=30,earlystop=5,transferlayer=1,inputweights=None,outputweights=None,forkinas=False):
    
    string = 'C:/Users/coffee/Desktop/ubisiteeeeeee/Ubiquitination.elm'
    
    # train dataset
    seq_list = read_fasta(string, 12)
    seq = pd.DataFrame(seq_list)
    trainX = seq.as_matrix()
    
    # train_pos  train_neg
    train_pos=trainX[np.where(trainX[:,0]==1)]
    train_neg=trainX[np.where(trainX[:,0]!=1)]
    train_pos=pd.DataFrame(train_pos)
    train_neg=pd.DataFrame(train_neg) 
    
    a=int(train_pos.shape[0]*0.9)
    b=train_neg.shape[0]-int(train_pos.shape[0]*0.1)
    train_pos_s=train_pos[0:a]
    train_neg_s=train_neg[0:b]
    
    # val dataset
    val_pos=train_pos[(a+1):];
    val_neg=train_neg[(b+1):];
    val_all=pd.concat([val_pos,val_neg])
    valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode)
   
    c = int(train_pos_s.shape[0]*0.6);
    d = train_neg_s.shape[0]-int(train_pos_s.shape[0]*0.4)
    train_pos_ss = train_pos_s[0:c]
    train_neg_ss = train_neg_s[0:d]
    
    # test dataset
    test_pos = train_pos_s[(c+1):]
    test_neg = train_neg_s[(d+1):]
    test_all = pd.concat([test_pos,test_neg])
    testX1,testY1 = convertRawToXY(test_all.as_matrix(),codingMode=codingMode)
    
    slength=int(train_pos_ss.shape[0]*srate); 
    nclass=int(train_neg_ss.shape[0]/slength); 
    
    for i in range(nb_epoch1):
        train_neg_ss=train_neg_ss.sample(train_neg_ss.shape[0]); #shuffle neg sample
        train_pos_sss=train_pos_ss.sample(slength) 
        for t in range(nclass):
            train_neg_sss=train_neg_ss[(slength*t):(slength*t+slength)];
            train_all=pd.concat([train_pos_sss,train_neg_sss])
            trainX1,trainY1 = convertRawToXY(train_all.as_matrix(),codingMode=codingMode) 
            if t==0:
                models,testX_t=construct_cnn(trainX1,trainY1,valX1,valY1,testX1,testY1,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t)
            else:
                models,testX_t=construct_cnn(trainX1,trainY1,valX1,valY1,testX1,testY1,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t,compilemodels=models)
                 
            if(outputweights is not None):
                models.save_weights(outputweights,overwrite=True)   
    acc, precision, sensitivity, specificity, MCC  = predict(models,testX_t,testY1)
  
    #evalResult.evaluate(predictions,predict_y,testX1,testY1)    

    