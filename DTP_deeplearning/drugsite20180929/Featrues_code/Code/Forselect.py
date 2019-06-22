import pickle,random
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from minepy import MINE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from Get_Samples import GetSample

class Feature_select():
    
    def mic(self,x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)
    
    def get_best_features(self, originalX, originalY, K = 5): 
        #print(originalX.shape[1])
        micscores  = []
        pvals = []
        for i in range(0,originalX.shape[1]):
            x = (originalX[:,i].T)
            #print((originalX[:,i].T).shape)
            micscore = self.mic(x, originalY)[0]
            micscores.append(micscore)
            pvals.append(self.mic(x, originalY)[1])
            
            
            print(i,'' ,micscore)
            
            #print(type(originalX.shape()))
            
            #print()
            newset = SelectKBest(f_classif, k=100).fit_transform(originalX, originalY)
        #newmicset = SelectKBest((np.array(micscores),np.array(pvals)), k=K).fit_transform(originalX, originalY)
        print(newset)
        print(newset.shape)
       
        return newset 
    
if __name__ == '__main__':
    sample = GetSample()
    X_train,y_train = sample.main(winn= 0,random_nag = True,rate = 1.0) 
    fselect = Feature_select()
    #fselect.mic(x, y)
    g = fselect.get_best_features(X_train,y_train, K = 576) 