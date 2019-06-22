import pickle,random
import numpy as np

class GetSample():
    
    
    def main(self,winn= 0,random_nag = False,rate = 1.0):
        pssm,topo,onehot,aaindex = self.getfeatures()      
        lablenum = self.getlabels()     
        labelIDslist = self.getlabelIDsList()  
        test_labelID,train_labelID = self.split_test_train(labelIDslist)
        #for winn in range(0,51):
        print('windows is ' + str(winn))
        train_label,train_feature = self.get_label_fealist(train_labelID,lablenum,pssm,topo,onehot,aaindex,win = winn)
        
        test_label,test_feature = self.get_label_fealist(test_labelID,lablenum,pssm,topo,onehot,aaindex,win = winn,array=True)

        #print(len(label),len(feature),len(feature))
        nagativelabel,nagativefeature,positivelabel,positivefeature = self.get_pos_nag( train_label,train_feature)  
        #kernel = 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
            
        X_train,y_train = self.getBacksamples(nagativelabel,nagativefeature,positivelabel,positivefeature,random_nag = random_nag,rate = rate)     
        
        X_train,y_train = self.getshufflesamples(X_train,y_train)
        X_test,y_test = self.getshufflesamples(test_feature,test_label)        
        return X_train,y_train,X_test,y_test 
        
        
    def getfeatures(self):


        picpssm= open('/home/luchang/TMP/Features/PSSM_pickle.pic','rb')
        pssm = pickle.load(picpssm)

        #psi= open('/home/luchang/TMP/Features/psiprednum.pickle','rb')
        #psipred = pickle.load(psi)    

        pica = open('/home/luchang/TMP/Features/Aaindex_pickle.pic','rb')
        aaindex = pickle.load(pica)

        pictopo = open('/home/luchang/TMP/Features/Topology_pickle.pic','rb')
        topo = pickle.load(pictopo)

        pic1hot = open('/home/luchang/TMP/Features/Onehot_pickle.pic','rb')
        onehot = pickle.load(pic1hot)

        return pssm,topo,onehot,aaindex#,psipred


    def getlabels(self):
        sPickle = open('/home/luchang/TMP/DATA/label_1151_pickle.pic','rb')
        lablenum = pickle.load(sPickle)   
        return lablenum


    def get_label_fealist(self,labelIDslist,lablenum,pssm,topo,onehot,aaindex,win = 0,array = False):
        label = []
        feature = []
        for each in labelIDslist:  
            feat = []
            for res_i in range(0,len(pssm[each])):  
                fea = []
                fea.extend(pssm[each][res_i])
                #fea.extend(topo[each][res_i]) # one wei topo feature
                #fea.extend(aaindex[each][res_i])
                #fea.extend(onehot[each][res_i])
                #fea.extend(psipred[each][res_i])

                feat.append(fea)
            if len(lablenum[each]) == len(feat):
                label.extend(lablenum[each])
                if win == 0:
                    feature.extend(feat)
                elif win > 0: 
                    win_fea = self.getwinfeature(feat,win)
                    feature.extend(win_fea)    
            else:
                pass


        arraylabel = np.array(label,dtype = int) 
        arrayfeature =  np.array(feature,dtype = float)  
        print(np.shape(arrayfeature),np.shape(arraylabel))
        if array == True:
            return arraylabel,arrayfeature 
        else:
            return label,feature

    def getwinfeature(self,feat,win):
        win_fea = [] 
        for h in range(len(feat)):       
            win_fea.append(self.getonesitewinfea(feat, win,h))
        #print(np.shape(np.array(win_fea)))
        return win_fea


    def getonesitewinfea(self,feat,win,res_pos):

        tab_zero = np.ndarray.tolist(np.zeros(len(feat[0])) )
        if win <= res_pos and  len(feat) - win -1 >= res_pos :
            onesite_fea = []
            for p in range(res_pos-win ,res_pos+win+1):
                onesite_fea.extend(feat[p])
            return onesite_fea  
        elif win > res_pos and len(feat) - win - 1 >= res_pos :
            #left short
            onesite_fea = []
            for p in range(win-res_pos):
                onesite_fea.extend(tab_zero)
            for p in range( 0,res_pos+win+1):
                onesite_fea.extend(feat[p])
            return onesite_fea
        elif win <= res_pos and len(feat) - 1 - win < res_pos :
            #right short
            onesite_fea = []
            for p in range( res_pos-win ,len(feat)):
                onesite_fea.extend(feat[p])        
            for p in range(win+res_pos - len(feat) + 1):
                onesite_fea.extend(tab_zero)
            return onesite_fea
        else: 
            #left and right short
            onesite_fea = []
            for p in range(win-res_pos):
                onesite_fea.extend(tab_zero) 
            for p in range( 0,len(feat)):
                onesite_fea.extend(feat[p])         
            for p in range(win+res_pos - len(feat) + 1):
                onesite_fea.extend(tab_zero)   
            return onesite_fea 

    def get_pos_nag(self,label,feature)  :
        positivelabel = []
        nagativelabel = []
        positivefeature = []
        nagativefeature = []
        for pni in range(0,len(label)):
            each = label[pni]
            if each == 1:
                positivelabel.append(each)
                positivefeature.append(feature[pni])
            else:
                nagativelabel.append(each)
                nagativefeature.append(feature[pni])

        print(len(nagativelabel))
        print(len(nagativefeature))
        print(len(positivelabel))
        print(len(positivefeature))    

        return nagativelabel,nagativefeature,positivelabel,positivefeature


    def getBacksamples(self,nagativelabel,nagativefeature,positivelabel,positivefeature,random_nag = True,rate = 1.0):
        if random_nag == True:

            train_fea = positivefeature
            train_fea.extend(random.sample(nagativefeature, int(rate*len(positivelabel))))
            train_label = positivelabel
            train_label.extend(nagativelabel[:int(rate*len(positivelabel))])
        else:
            train_fea = positivefeature
            train_fea.extend(nagativefeature[0: int(rate*len(positivelabel))])
            train_label = positivelabel
            train_label.extend(nagativelabel[:int(rate*len(positivelabel))])        
        return train_fea,train_label


    def getshufflesamples(self,fea,label):

        X1 = np.array(fea,dtype = float)
        y1 = np.array(label,dtype = int)  

        r = np.random.permutation(len(y1)) 

        X = X1[r,:]
        Y = y1[r]    
        return X,Y

    def getlabelIDsList(self):
        sPickle = open('/home/luchang/TMP/DATA/ID_List_1151_Pickle.pic','rb')
        labelIDslist = pickle.load(sPickle)   
        print(len(labelIDslist))
        return labelIDslist    
    
    def split_test_train(self,labelIDslist):
        #print(len(labelIDslist))
        test_labelID = []
        train_labelID = []
        train = open('/home/luchang/TMP/DATA/ID_train.pic','rb')
        trainlabelIDs = pickle.load(train) 
        
        test = open('/home/luchang/TMP/DATA/ID_test.pic','rb')
        testlabelIDs = pickle.load(test)     
        
        for eitem in trainlabelIDs:
            if eitem in labelIDslist:
                train_labelID.append(eitem)
        print(len(train_labelID))
        
        for each in testlabelIDs:
            if each in labelIDslist:
                test_labelID.append(each)
        print(len(test_labelID))
        
        return test_labelID,train_labelID
    
        ##print(len(labelIDslist))
        ##print(len(labelIDs))
        ##print(len(test_label))
        ##print(len(train_label))