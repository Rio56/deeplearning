# -*- coding: utf-8 -*-
"""
# @Time    : 2018/6/8 23:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
from methods_2.DTP2_data_cls import DTP_prepare_data

# from "D:\\GitHub\\DTP_deeplearning\\DTP_deeplearning\\".methods.DTP_set_Net import zyh_CNN
from methods_2.DTP2_net_cls import DTP_set_net
from methods_2.DTP_DProcess import convertRawToXY
from methods_2.DTP2_set_GPU import set_GPU
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
from methods_2.DTP2_evaluate_cls import DTP_evaluate_modle

def start_training_2(system,fildername,feature_type):
    system =system
    fildername = fildername
    Data_prepare = DTP_prepare_data()
    Dp_net = DTP_set_net()
    feature_type = feature_type
    train_x, train_y = Data_prepare.load_data("train",feature_type ,n_to_p_rate=1, random_rate=0.8, system=system)
    val_x, val_y = Data_prepare.load_data("val",feature_type ,n_to_p_rate=1, random_rate=0.8, system=system)
    #test_x, test_y = Data_prepare.load_data("test", n_to_p_rate=1, random_rate=0.8, system="windows")

    
    accuracys_final = {'train_loss1': [], 'train_acc1': [], 'val_loss2': [], 'val_acc2': []}
    models, accuracys = Dp_net.zyh_CNN_mnist(train_x, train_y, val_x, val_y,compiletimes=0, forkinas=False, transferlayer=1, compilemodels=None,
                earlystop=None, nb_epoch=10, fildername=fildername)

    for item in range(0, 1000):
        train_x, train_y = Data_prepare.load_data("train", feature_type,n_to_p_rate=1, random_rate=0.8, system=system)
        val_x, val_y = Data_prepare.load_data("val", feature_type,n_to_p_rate=1, random_rate=0.8, system=system)
        models, accuracys = Dp_net.zyh_CNN_mnist(train_x, train_y, val_x, val_y, compiletimes=item, transferlayer=1, forkinas=1, compilemodels=models,
                                    fildername=fildername)
        
        eva_modle = DTP_evaluate_modle()
        


        img_rows , img_cols = val_x.shape[1], val_x.shape[2]
        print(img_rows , img_cols)

        #"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
        val_x = val_x.reshape(val_x.shape[0], img_rows, img_cols, 1)

        #more reshaping
        val_x = val_x.astype('float32')
        pre_score, pre, rec, SN, SP, f1, mcc, roc_auc = eva_modle.evaluate_model(models, val_x, val_y, item, "__", fildername)
        accuracys_final['train_loss1'].append(accuracys['train_loss1'])
        accuracys_final['train_acc1'].append(accuracys['train_acc1'])
        accuracys_final['val_loss2'].append(accuracys['val_loss2'])
        accuracys_final['val_acc2'].append(accuracys['val_acc2'])
        eva_modle.print_final_acc_loss(accuracys_final, item, "__", fildername)
        print("pre_score, pre, rec, SN, SP, f1, mcc, roc_auc")
        print(pre_score, pre, rec, SN, SP, f1, mcc, roc_auc)
        
if __name__ == "__main__":
    set_GPU(0)
    starttime = datetime.datetime.now()
    feature_type = "one_hot"
    #feature_type = "phy_che"    
    start_training_2("linux","result",feature_type)
    
    endtime = datetime.datetime.now()
    print("totaltime = " + str(endtime - starttime))
