# -*- coding: utf-8 -*-
"""
# @Time    : 2018/6/8 23:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import os
import datetime

import pandas as pd
import numpy as np
import pickle
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score


class DTP_evaluate_modle(object):    
    def evaluate_model(self,modelname, testX, testY, i, type, fildername):
        cnn = modelname
        # cnn = models.load_model('%d-merge.h5' % i, {'isru': isru, 'pearson_r': pearson_r})
        #  ############### test ##########################
        pre_score = cnn.evaluate(testX, testY, batch_size=4096, verbose=0)
        print(pre_score)
    
        #fileX = open('./' + fildername + '/pre_score_%d.pickle' % i, 'wb')
        #pickle.dump(pre_score, fileX, protocol=4)
        #fileX.close()
    
        # 最后做对比图写出来
        #  ######### Print Precision and Recall ##########
        pred_proba = cnn.predict(testX, batch_size=2048)
        pred_score = pred_proba[:, 1]
        true_class = testY[:, 1]
    
        precision, recall, _ = precision_recall_curve(true_class, pred_score)
        average_precision = average_precision_score(true_class, pred_score)
    
        fpr, tpr, thresholds = roc_curve(true_class, pred_score)
        roc_auc = auc(fpr, tpr)
        # print(pred_score)
        # 0.6 is zyh set
        for index in range(len(pred_score)):
            if pred_score[index] > 0.6:
                pred_score[index] = 1
            else:
                pred_score[index] = 0
    
        mcc = matthews_corrcoef(true_class, pred_score)
    
        plt.figure()
        plt.step(recall, precision, color='navy', where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.grid(True)
        plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        #plt.savefig('./' + fildername + '/Precision-Recall_%d.png' % i)
    
        #  ################# Print ROC####################
    
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Inception ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        #plt.savefig('./' + fildername + '/ROC_%d.png' % i)
        SN, SP = self.performance(true_class, pred_score)
        pre = precision_score(y_true=true_class, y_pred=pred_score)
        rec = recall_score(y_true=true_class, y_pred=pred_score)
        f1 = f1_score(y_true=true_class, y_pred=pred_score)
    
        # Sn和recall是同一个值
        return pre_score, pre, rec, SN, SP, f1, mcc, roc_auc
    
    
    def performance(self,labelArr, predictArr):
        TP = 0.
        TN = 0.
        FP = 0.
        FN = 0.
        for i in range(len(labelArr)):
            if labelArr[i] == 1 and predictArr[i] == 1:
                TP += 1.
            if labelArr[i] == 1 and predictArr[i] == 0:
                FN += 1.
            if labelArr[i] == 0 and predictArr[i] == 1:
                FP += 1.
            if labelArr[i] == 0 and predictArr[i] == 0:
                TN += 1.
        SN = TP / (TP + FN)
        SP = TN / (FP + TN)
        return SN, SP
    
    def print_final_acc_loss(self,accuracys_final, i, type, fildername):
        #  ######### Print Loss Map ##########
        plt.figure()
    
        plt.plot(accuracys_final['train_loss1'])
        plt.plot(accuracys_final['val_loss2'])
        plt.plot(accuracys_final['train_acc1'])
        plt.plot(accuracys_final['val_acc2'])
        plt.title('LOSS_ACC:times %d' % i)
        plt.ylim([0, 1.0])
        plt.ylabel('loss_acc')
        plt.xlabel('times')
        plt.grid(True)
        plt.legend(['train_loss1', 'val_loss2', 'train_acc1', 'val_acc2'], loc='upper left')
        plt.savefig('./' + fildername + '/' + str(type) + 'loss_acc%d.png' % i)    
        
        

    def print_loss_acc(self,fitHistory, i, type, fildername="fildername"):
        #  ######### Print Loss Map ##########
        plt.figure()
        plt.plot(fitHistory.history['loss'])
        plt.plot(fitHistory.history['val_loss'])
        plt.plot(fitHistory.history['acc'])
        plt.plot(fitHistory.history['val_acc'])
        # plt.title('size:%d' % size)
        plt.title('LOSS:times %d' % i)
        plt.ylim([0, 1.0])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(['train_loss', 'validation_loss', 'train_acc', 'validation_acc'], loc='upper left')
        plt.savefig('./' + fildername + '/' + str(type) + '%d.png' % i)
    
        #  ############### final ################
        loss1 = fitHistory.history['loss'][-1]
        acc1 = fitHistory.history['acc'][-1]
        loss2 = fitHistory.history['val_loss'][-1]
        acc2 = fitHistory.history['val_acc'][-1]
    
        return loss1, acc1, loss2, acc2
    
    
    def print_acc(self,fitHistory, i, type, fildername):
        #  ######### Print Loss Map ##########
        plt.figure()
        plt.plot(fitHistory.history['acc'])
        plt.plot(fitHistory.history['val_acc'])
        # plt.title('size:%d' % size)
        plt.title('ACC:times %d' % i)
        plt.ylim([0, 1.0])
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('./' + fildername + '/' + str(type) + '%d.png' % i)
    
        #  ############### final ################
        loss1 = fitHistory.history['loss'][0]
        acc1 = fitHistory.history['acc'][0]
        loss2 = fitHistory.history['val_loss'][0]
        acc2 = fitHistory.history['val_acc'][0]
    
        return loss1, acc1, loss2, acc2
        
        
    def print_loss(self,fitHistory, i, type, fildername):
        #  ######### Print Loss Map ##########
        plt.figure()
        plt.plot(fitHistory.history['loss'])
        plt.plot(fitHistory.history['val_loss'])
        # plt.title('size:%d' % size)
        plt.title('LOSS:times %d' % i)
        plt.ylim([0, 1.0])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('./' + fildername + '/' + str(type) + '%d.png' % i)
    
        #  ############### final ################
        loss1 = fitHistory.history['loss'][0]
        acc1 = fitHistory.history['acc'][0]
        loss2 = fitHistory.history['val_loss'][0]
        acc2 = fitHistory.history['val_acc'][0]
    
        return loss1, acc1, loss2, acc2    
        
        
        
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    eva_modle = DTP_evaluate_modle()
    
    
    endtime = datetime.datetime.now()
    print("totaltime = " + str(endtime - starttime))
