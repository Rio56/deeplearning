from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
import pickle,random,sklearn,math
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,matthews_corrcoef,roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,auc
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class LeaveoneNfold():
    
    def Divide_sample(self,X,y,fun = 'svm'):
        loo = LeaveOneOut()
        se = Fit_model()
        y_true = []
        y_pred = []
            
        for train_index, test_index in loo.split(X):
            n = []
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]  
            if fun == 'svm':
                predict_y = se.svmpredict(X_train, X_test,y_train, y_test)   
            elif fun == 'rf':
                predict_y = se.rfpredict(X_train, X_test,y_train, y_test)    
            elif fun == 'kn':
                predict_y = se.knpredict(X_train, X_test,y_train, y_test)     
            elif fun == 'MLP':
                predict_y = se.MLPpredict(X_train, X_test,y_train, y_test)                  
            #model(X_train, X_test,y_train, y_test,fun =fun)
            y_true.append(y_test[0])
            y_pred.append(predict_y[0])
            #print(y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return y_true,y_pred
    
class fold10():
    
    def foldmodel(self,X,y,fun = 'svm'):

        se = Fit_model()
        y_true = []
        y_pred = []
        cv = StratifiedKFold(y, n_folds=10)       
        for i, (train_index, test_index) in enumerate(cv):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]  
            if fun == 'svm':
                predict_y = se.svmpredict(X_train, X_test,y_train, y_test)   
            elif fun == 'rf':
                predict_y = se.rfpredict(X_train, X_test,y_train, y_test)    
            elif fun == 'kn':
                predict_y = se.knpredict(X_train, X_test,y_train, y_test)    
            elif fun == 'MLP':
                predict_y = se.MLPpredict(X_train, X_test,y_train, y_test)                    
            
            #model(X_train, X_test,y_train, y_test,fun =fun)
            y_true.extend(np.ndarray.tolist(y_test))
            y_pred.extend(np.ndarray.tolist(predict_y))
            #print(y_true)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return y_true,y_pred
    
            
class Fit_model():

    def svmpredict(self,X_train, X_test,y_train, y_test):
        #auc
        #kernel = 'linear', 'poly', 'rbf', 'sigmoid'
        clf= svm.SVR()
        #clf= svm.SVC(C=10, gamma=0.1, kernel='rbf',probability=True)
        clf.fit(X_train,y_train)    
        y_predict =clf.predict(X_test)    

        return y_predict
    
    def rfpredict(self,X_train, X_test,y_train, y_test):
        #auc
        clf = RandomForestClassifier()
        #clf = RandomForestClassifier(n_estimators=180 , criterion="gini")
        clf.fit( X_train,y_train)
        y_predict = clf.predict(X_test)        
        return y_predict
    
    def knpredict(self,X_train, X_test,y_train, y_test):
        #auc
        knn = KNeighborsClassifier()
        #knn = KNeighborsClassifier(n_neighbors=100,weights='distance')
        knn.fit( X_train,y_train)
        y_predict = knn.predict(X_test)     
        return y_predict
    
    
    def MLPpredict(self,X_train, X_test,y_train, y_test):
        mlp = MLPClassifier()
        mlp.fit( X_train,y_train)
        y_predict = mlp.predict(X_test)
        return y_predict  


class GetMetrics():
    
    def indexes(self,y_true, y_pred):
        #Compute confusion matrix
        print("===metrics ===")
        cm = confusion_matrix(y_true, y_pred)
        
        mcc = matthews_corrcoef(y_true,y_pred)
        
        acc = accuracy_score(y_true,y_pred)
        
        pre = precision_score(y_true, y_pred)
        
        rec = recall_score(y_true, y_pred)
        
        f1_s = f1_score(y_true,y_pred)
       
        return cm,acc,mcc,pre,rec,f1_s    
    
    
    def AUC(self,y_test, y_pred):
        print("===compute auc ===")
        #compute the auc
        aucscore = roc_auc_score(y_test,y_pred)
        return aucscore   


  
        
if __name__ == '__main__':     
    
        index = GetMetrics()
        lo = fold10()
                
        X_train,y_train = sample.main(winn= 0,random_nag = True,rate = 1.0) 
        y_true, y_pred = lo.foldmodel(X_train,y_train,fun = 'rf')
        cm,acc,mcc,pre,rec,f1_s   = index.indexes(y_true, y_pred)    
        aucscore = index.AUC(y_true, y_pred)
        print(cm,acc,mcc,pre,rec,f1_s ,aucscore )        
    
