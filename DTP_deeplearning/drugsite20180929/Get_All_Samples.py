import pickle,random
import numpy as np

class GetallSample():
    
    
    def main(self,winn= 0,random_nag = False,rate = 1.0):
     
        labelIDslist = self.getlabelIDsList()  
        pssm,topo,onehot = self.getfeatures()
        lablenum = self.getlabels()        
        train_data = self.get_label_fealist(labelIDslist,lablenum,pssm,topo,onehot)     
        
        #X_train,y_train = self.getshufflesamples(X_train,y_train)     
        return train_data 
        
        
    def getfeatures(self):  
            
        picpssm= open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/PSSM_normalized_3234..pic','rb')
        pssm = pickle.load(picpssm)
        
        #pica = open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/Aaindex_pickle.pic','rb')
        #aaindex = pickle.load(pica)
        
        pictopo = open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/Topology_pickle.pic','rb')
        topo = pickle.load(pictopo)
        
        pic1hot = open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/Onehot_pickle.pic','rb')
        onehot = pickle.load(pic1hot)

        return pssm,topo,onehot#,aaindex,psipred
    
    
    def getlabels(self):
        sPickle = open('/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/po_ne_specific/labeldicts.pic','rb')
        labeldicts = pickle.load(sPickle)   
        return labeldicts
    
    
    def get_label_fealist(self,labelIDslist,lablenum,pssm,topo,onehot):
        label = []
        feature = []
        for each in labelIDslist:  
            feat = []
            for res_i in range(0,len(pssm[each])):  
                fea = []
                fea.extend(pssm[each][res_i])
                fea.extend(topo[each][res_i]) # one wei topo feature
                #fea.extend(psipred[each][res_i])
                #fea.extend(aaindex[each][res_i])
                fea.extend(onehot[each][res_i])
            feat.append(fea)
            label.extend(lablenum[each])
        
        
        arraylabel = np.array(label,dtype = int) 
        arrayfeature =  np.array(feature,dtype = float)  
        
        train_all = np.concatenate((a,b),axis=1)


        return train_all 

    
    def getshufflesamples(self,fea,label):
     
        X1 = np.array(fea,dtype = float)
        y1 = np.array(label,dtype = int)  
    
        r = np.random.permutation(len(y1)) 
        
        X = X1[r,:]
        Y = y1[r]    
        return X,Y
    
    def getlabelIDsList(self):
        
        #lines = open('/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/po_ne_specific/seq_0.3cutoff.fasta','r').readlines()
        lines = open('/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/po_ne_specific/seq_cut_0.9.fasta','r').readlines()
        idlist = []
        for i in range(len(lines)):
            line = lines[i]
            if line[0] == '>':
                idlist.append(line[0:-1])
        return idlist
