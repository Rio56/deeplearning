
import numpy as np
import os
from keras.utils import to_categorical

def format(f, n):
    if round(f)==f:
        m = len(str(f))-1-n
        if f/(10**m) ==0.0:
            return f
        else:
            return float(int(f)/(10**m)*(10**m))
    return round(f, n - len(str(int(f)))) if len(str(f))>n+1 else f

window_length = 19

dict = {'C':0, 'D':1, 'S':2, 'Q':3, 'K':4,
        'I':5, 'P':6, 'T':7, 'F':8, 'N':9,
        'G':10, 'H':11, 'L':12, 'R':13, 'W':14,
        'A':15, 'V':16, 'E':17, 'Y':18, 'M':19}

def train_data_pre_processing():
    train_fasta = open("train_data/train.fasta")
    line = train_fasta.readline()
    pdb_id = ""
    x_train = []
    y_train = []    
    while line:
        codelist = []
        if(line[0] == ">"):
            pdb_id = line[1:7]
            line = train_fasta.readline()
            continue
        #print(pdb_id)
        #-------- one-hot encoded (len * 20)----------#
        for i in line:
            if(i != "\n"):
                code = dict[i.upper()]
                codelist.append(code)
        data = np.array(codelist)
        #print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        if(encoded.shape[1] <20):
            column = np.zeros([encoded.shape[0],20-encoded.shape[1]],int)
            encoded = np.c_[encoded,column]
        #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        #print(encoded)
        #print(encoded.shape)
        #-------- one-hot encoded (len * 20)----------#
        
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        code = encoded
        length = code.shape[0]
        feature = code.shape[1]
        #print(length)
        #print(feature)
        noSeq = np.zeros([length,1],int)
        code = np.c_[code,noSeq]
        
        t = int((window_length - 1) / 2)
        
        code = np.r_[np.c_[np.zeros([t,20],int),np.ones(t,int)],code]        
        code = np.r_[code,np.c_[np.zeros([t,20],int),np.ones(t,int)]]
        #print(code.shape)
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        
        #-------- sliding window (window_length * 21)---------#
        length = code.shape[0]
        feature = code.shape[1]
        top = 0
        buttom = window_length
        while(buttom <= length):
            #print(code[top:buttom]) 
            #print(code[top:buttom].shape) # 17 * 21
            x_train.append(code[top:buttom])            
            top += 1
            buttom += 1
        #print(len(window_list))
        
        #-------- sliding window (window_length * 21)---------#
        
        #-------- label mapping ---------#
        label_list = []
        line = open("typetmfiles/" + pdb_id + ".type")
        label = line.readline()     
        while label:
            num = float(label[0:len(label)])
            num = round(num,2)
            y_train.append(num)          
            #print(label[0:len(label)-1])
            label = line.readline()
        #print(label_list)
        #print(len(label_list))
        #-------- label mapping ---------#
        
        line = train_fasta.readline()    
        
    #print(len(x_train))
    x_train = np.array(x_train)
    print(x_train.shape)
    #print(y_train)
    y_train = np.array(y_train)
    print(y_train.shape)    
    np.save("train_data/x_train_winlen_" + str(window_length) + ".npy", x_train, int)
    np.save("train_data/y_train_winlen_" + str(window_length) + ".npy", y_train, int)

def valid_data_pre_processing():
    train_fasta = open("valid_data/valid.fasta")
    line = train_fasta.readline()
    pdb_id = ""
    x_train = []
    y_train = []    
    while line:
        codelist = []
        if(line[0] == ">"):
            pdb_id = line[1:7]
            line = train_fasta.readline()
            continue
        #print(pdb_id)
        #-------- one-hot encoded (len * 20)----------#
        for i in line:
            if(i != "\n"):
                code = dict[i.upper()]
                codelist.append(code)
        data = np.array(codelist)
        #print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        if(encoded.shape[1] <20):
            column = np.zeros([encoded.shape[0],20-encoded.shape[1]],int)
            encoded = np.c_[encoded,column]
        #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        #print(encoded)
        #print(encoded.shape)
        #-------- one-hot encoded (len * 20)----------#
        
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        code = encoded
        length = code.shape[0]
        feature = code.shape[1]
        #print(length)
        #print(feature)
        noSeq = np.zeros([length,1],int)
        code = np.c_[code,noSeq]
        
        t = int((window_length - 1) / 2)
        
        code = np.r_[np.c_[np.zeros([t,20],int),np.ones(t,int)],code]        
        code = np.r_[code,np.c_[np.zeros([t,20],int),np.ones(t,int)]]
        #print(code.shape)
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        
        #-------- sliding window (window_length * 21)---------#
        length = code.shape[0]
        feature = code.shape[1]
        top = 0
        buttom = window_length
        while(buttom <= length):
            #print(code[top:buttom]) 
            #print(code[top:buttom].shape) # 17 * 21
            x_train.append(code[top:buttom])            
            top += 1
            buttom += 1
        #print(len(window_list))
        
        #-------- sliding window (window_length * 21)---------#
        
        #-------- label mapping ---------#
        label_list = []
        line = open("typetmfiles/" + pdb_id + ".type")
        label = line.readline()     
        while label:
            num = float(label[0:len(label)])
            num = round(num,2)
            y_train.append(num)          
            #print(label[0:len(label)-1])
            label = line.readline()
        #print(label_list)
        #print(len(label_list))
        #-------- label mapping ---------#
        
        line = train_fasta.readline()    
        
    #print(len(x_train))
    x_train = np.array(x_train)
    print(x_train.shape)
    #print(y_train)
    y_train = np.array(y_train)
    print(y_train.shape)    
    np.save("valid_data/x_valid_winlen_" + str(window_length) + ".npy", x_train, int)
    np.save("valid_data/y_valid_winlen_" + str(window_length) + ".npy", y_train, int)
    
def test_data_pre_processing():
    train_fasta = open("test_data/test.fasta")
    line = train_fasta.readline()
    pdb_id = ""
    x_train = []
    y_train = []
    while line:
        codelist = []
        if(line[0] == ">"):
            pdb_id = line[1:7]
            line = train_fasta.readline()
            continue
        #print(pdb_id)
        #-------- one-hot encoded (len * 20)----------#
        for i in line:
            if(i != "\n"):
                code = dict[i.upper()]
                codelist.append(code)
        data = np.array(codelist)
        #print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        if(encoded.shape[1] <20):
            column = np.zeros([encoded.shape[0],20-encoded.shape[1]],int)
            encoded = np.c_[encoded,column]
        #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        #print(encoded)
        #print(encoded.shape)
        #-------- one-hot encoded (len * 20)----------#
        
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        code = encoded
        length = code.shape[0]
        feature = code.shape[1]
        #print(length)
        #print(feature)
        noSeq = np.zeros([length,1],int)
        code = np.c_[code,noSeq]
        
        t = int((window_length - 1) / 2)
        
        code = np.r_[np.c_[np.zeros([t,20],int),np.ones(t,int)],code]        
        code = np.r_[code,np.c_[np.zeros([t,20],int),np.ones(t,int)]]
        #print(code.shape)
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        
        #-------- sliding window (window_length * 21)---------#
        length = code.shape[0]
        feature = code.shape[1]
        top = 0
        buttom = window_length
        while(buttom <= length):
            #print(code[top:buttom]) 
            #print(code[top:buttom].shape) # 17 * 21
            x_train.append(code[top:buttom])            
            top += 1
            buttom += 1
        #print(len(window_list))
        
        #-------- sliding window (window_length * 21)---------#
        
        #-------- label mapping ---------#
        label_list = []
        line = open("typetmfiles/" + pdb_id + ".type")
        label = line.readline()     
        while label:
            num = float(label[0:len(label)])
            num = round(num,2)
            y_train.append(num)          
            #print(label[0:len(label)-1])
            label = line.readline()
        #print(label_list)
        #print(len(label_list))
        #-------- label mapping ---------#
        
        line = train_fasta.readline()    
        
    #print(len(x_train))
    x_train = np.array(x_train)
    print(x_train.shape)
    #print(y_train)
    y_train = np.array(y_train)
    print(y_train.shape)
    np.save("test_data/x_test_winlen_" + str(window_length) + ".npy", x_train, int)
    np.save("test_data/y_test_winlen_" + str(window_length) + ".npy", y_train, int)
    
    
def comp_data_pre_processing():
    train_fasta = open("comp_data/comp.fasta")
    line = train_fasta.readline()
    pdb_id = ""
    x_train = []
    y_train = []
    while line:
        codelist = []
        if(line[0] == ">"):
            pdb_id = line[1:7]
            line = train_fasta.readline()
            continue
        #print(pdb_id)
        #-------- one-hot encoded (len * 20)----------#
        for i in line:
            if(i != "\n"):
                code = dict[i.upper()]
                codelist.append(code)
        data = np.array(codelist)
        #print('Shape of data (BEFORE encode): %s' % str(data.shape))
        encoded = to_categorical(data)
        if(encoded.shape[1] <20):
            column = np.zeros([encoded.shape[0],20-encoded.shape[1]],int)
            encoded = np.c_[encoded,column]
        #print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
        #print(encoded)
        #print(encoded.shape)
        #-------- one-hot encoded (len * 20)----------#
        
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        code = encoded
        length = code.shape[0]
        feature = code.shape[1]
        #print(length)
        #print(feature)
        noSeq = np.zeros([length,1],int)
        code = np.c_[code,noSeq]
        
        t = int((window_length - 1) / 2)
        
        code = np.r_[np.c_[np.zeros([t,20],int),np.ones(t,int)],code]        
        code = np.r_[code,np.c_[np.zeros([t,20],int),np.ones(t,int)]]
        #print(code.shape)
        #-------- noseq encoded (len + window_length - 1) * 21----------#
        
        #-------- sliding window (window_length * 21)---------#
        length = code.shape[0]
        feature = code.shape[1]
        top = 0
        buttom = window_length
        while(buttom <= length):
            #print(code[top:buttom]) 
            #print(code[top:buttom].shape) # 17 * 21
            x_train.append(code[top:buttom])            
            top += 1
            buttom += 1
        #print(len(window_list))
        
        #-------- sliding window (window_length * 21)---------#
        
        #-------- label mapping ---------#
        label_list = []
        line = open("tmfiles/" + pdb_id)
        label = line.readline()     
        while label:
            num = float(label[0:len(label)])
            num = round(num,2)
            y_train.append(num)          
            #print(label[0:len(label)-1])
            label = line.readline()
        #print(label_list)
        #print(len(label_list))
        #-------- label mapping ---------#
        
        line = train_fasta.readline()    
        
    #print(len(x_train))
    x_train = np.array(x_train)
    print(x_train.shape)
    #print(y_train)
    y_train = np.array(y_train)
    print(y_train.shape)
    np.save("comp_data/x_comp_winlen_" + str(window_length) + ".npy", x_train, int)
    np.save("comp_data/y_comp_winlen_" + str(window_length) + ".npy", y_train, int)

#train_data_pre_processing()   
#valid_data_pre_processing()    
test_data_pre_processing()    
#comp_data_pre_processing()

#predict_data = open("predict_data/predict.fasta")

#valid_fasta = open("valid_data/valid.fasta")
#test_fasta = open("test_data/test.fasta")
#line = train_fasta.readline()
#while line:
    
    #if(line[0] == ">"):
        #line = predict.readline()
        #continue
    
    
    
    #line = predict.readline()