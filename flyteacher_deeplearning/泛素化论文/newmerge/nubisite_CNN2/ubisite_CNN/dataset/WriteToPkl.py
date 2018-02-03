import numpy as np
import pickle as pickle

def validWrite():
    with open('validing300.txt','r') as fp:
        j = -1
        valid_array = np.zeros((2000,300))
        for line in fp:
            j = j+1
            k = -1
            line = line.split('\t')
            for i in range(len(line)):
                k+=1
                valid_array[j][k] = float(line[i])#round(float(line[i]),4)
    with open('validing_label.txt','r') as fp:
        valid_label = np.zeros((2000))
        j = -1
        for line in fp:
            j += 1
            valid_label[j] = float(line)
    (x,y) = [valid_array,valid_label]
    valid_list = (x,y)
    return valid_list
    
def testWrite():
    with open('testing.txt','r') as fp:
        j = -1
        test_array = np.zeros((2400,900))
        for line in fp:
            j = j+1
            k = -1
            line = line.split(' ')
            for i in range(len(line)):
                k+=1
                test_array[j][k] = float(line[i])#round(float(line[i]),4)
                
    with open('testing_label.txt','r') as fp:
        test_label = np.zeros((2400))
        j = -1
        for line in fp:
            j += 1
            test_label[j] = float(line)
    (x,y) = [test_array,test_label]
    test_list = (x,y)
    return test_list

def trainWrite():
    with open('training.txt','r') as fp:
        j = -1
        train_array = np.zeros((600,900))
        for line in fp:
            j = j+1
            k = -1
            line = line.split(' ')
            for i in range(len(line)):
                k+=1
                train_array[j][k] = float(line[i])#round(float(line[i]),4)
    with open('training_label.txt','r') as fp:
        train_label = np.zeros((600))
        j = -1
        for line in fp:
            j += 1
            train_label[j] = float(line)
    
    (x,y) = [train_array,train_label]
    train_list = (x,y)
    
    test_list = testWrite()
    #valid_list = validWrite()
    
    (x,y) = [train_list,test_list]
    dataset = (x,y)
    print(dataset)
    
    fw = open('test1.pkl','wb')
    pickle.dump(dataset,fw)
    fw.close()
    
if __name__=="__main__":
    trainWrite()