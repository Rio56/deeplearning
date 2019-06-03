# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
"""from methods.DTP_set_Net import zyh_CNN
from methods.DTP_DProcess import convertRawToXY
from methods.DTP_prepare_data import change_data_into_ten_fold"""

#from "D:\\GitHub\\DTP_deeplearning\\DTP_deeplearning\\".methods.DTP_set_Net import zyh_CNN
from methods.DTP_set_Net import zyh_CNN
from methods.DTP_DProcess import convertRawToXY
from methods.DTP_get_data import change_data_into_ten_fold
from methods.DTP_set_GPU import set_GPU
import os

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score, f1_score
def start_training(train_pos_data, train_neg_data, val_pos_data, val_neg_data,srate = 0.8,epoch = 3, codingMode=0):
	"""
	:argument
	:return:
	"""
	print("!")
	
	val_all = pd.concat([val_pos_data, val_neg_data])
	print(val_all)
	valX1, valY1 = convertRawToXY(val_all.as_matrix(), codingMode=codingMode)
	print(valX1)
	print(valY1)
	change_data_into_ten_fold(valX1,valY1)
	
	
	slength = int(train_pos_data.shape[0] * srate);  # srate = 0.8
	nclass = int(train_neg_data.shape[0] / slength);  # find one pos vs ? neg.
	
	for epoch_times in range(epoch):
		train_neg_data = train_neg_data.sample(train_neg_data.shape[0]);  # shuffle neg sample
		train_pos_data = train_pos_data.sample(slength)
		
		for nclass_times in range(nclass):
			train_neg_data = train_neg_data[(slength * nclass_times):(slength * nclass_times + slength)];
			train_all = pd.concat([train_pos_data, train_neg_data])
			print(train_all)
			trainX1, trainY1 = convertRawToXY(train_all.as_matrix(), codingMode=codingMode)
			print(trainX1)
			#trainX1 = trainX1[(slength * nclass_times):(slength * nclass_times + slength)]
			
			if epoch_times == 0 and nclass_times == 0:
				models = zyh_CNN(trainX1, trainY1, valX1, valY1, compiletimes=nclass_times)
			else:
				models = zyh_CNN(trainX1, trainY1, valX1, valY1, compiletimes=nclass_times, compilemodels=models)
	pass

def evaluate_model(modelname, testX, testY, i, type, fildername):
	cnn = modelname
	# cnn = models.load_model('%d-merge.h5' % i, {'isru': isru, 'pearson_r': pearson_r})
	#  ############### test ##########################
	pre_score = cnn.evaluate(testX, testY, batch_size=4096, verbose=0)
	print(pre_score)

	fileX = open('./'+fildername+'/pre_score%d.pickle' % i, 'wb')
	pickle.dump(pre_score, fileX, protocol=4)
	fileX.close()

	# 最后做对比图写出来
	#  ######### Print Precision and Recall ##########
	pred_proba = cnn.predict(testX, batch_size=2048)
	pred_score = pred_proba[:, 1]
	true_class = testY[:, 1]

	precision, recall, _ = precision_recall_curve(true_class, pred_score)
	average_precision = average_precision_score(true_class, pred_score)

	fpr, tpr, thresholds = roc_curve(true_class, pred_score)
	roc_auc = auc(fpr, tpr)
	#print(pred_score)
        #0.6 is zyh set
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
	plt.savefig('./'+fildername+'/Precision-Recall%d.png' % i)

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
	plt.savefig('./'+fildername+'/ROC %d.png' % i)
	SN, SP = performance(true_class, pred_score)
	pre = precision_score(y_true=true_class, y_pred=pred_score)
	rec = recall_score(y_true=true_class, y_pred=pred_score)
	f1 = f1_score(y_true=true_class, y_pred=pred_score)

	# Sn和recall是同一个值
	return pre_score, pre, rec, SN, SP, f1, mcc, roc_auc

def performance(labelArr, predictArr):
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

def using_one_of_ten_fold_crossing(one_of_ten_fold_file_path,x_fold_type,fildername):
	path = one_of_ten_fold_file_path
	dirs = os.listdir(path)
	train_neg = []
	train_pos = []
	val_neg = []
	val_pos = []	
	accuracys_final = {'train_loss1':[], 'train_acc1':[], 'val_loss2':[], 'val_acc2':[]}
	for dir in dirs:
		if "neg" in dir and "train" in dir :
			train_neg.append(dir)
			pass
		if "pos" in dir and "train" in dir:
			train_pos.append(dir)
		if "neg" in dir and "test" in dir :
			val_neg.append(dir)
			pass
		if "pos" in dir and "test" in dir:
			val_pos.append(dir)	
	
	train_neg_data = np.load(path + train_neg[0])
	train_pos_data = np.load(path + train_pos[0])

	val_neg_data = np.load(path + val_neg[0])
	val_pos_data = np.load(path + val_pos[0])
	"""
	print(len(train_neg_data))
	print(len(train_pos_data))
	print(len(val_neg_data))
	print(len(val_pos_data))"""

	train_neg_data = pd.DataFrame(train_neg_data)
	train_pos_data = pd.DataFrame(train_pos_data)
	val_neg_data = pd.DataFrame(val_neg_data)
	val_pos_data = pd.DataFrame(val_pos_data)


	codingMode = 0

	train_all = pd.concat([train_pos_data, train_neg_data])
	trainX1, trainY1 = convertRawToXY(train_all.as_matrix(), codingMode=codingMode)		

	val_all = pd.concat([val_pos_data, val_neg_data])
	#print(val_all)
	valX1, valY1 = convertRawToXY(val_all.as_matrix(), codingMode=codingMode)
	#print(valX1)
	#print(valY1)
	
	models,accuracys = zyh_CNN(trainX1, trainY1, valX1, valY1, compiletimes=0,transferlayer=1,forkinas=1,fildername = fildername)
	
	compile_flag = 0
	
	while compile_flag < 300:
		for nclass_times in range(len(train_neg)):
			models,accuracys =run_model(nclass_times,compile_flag,one_of_ten_fold_file_path,fildername,models,train_neg,train_pos,val_neg,val_pos)
	
		pre_score, pre, rec, SN, SP, f1, mcc, roc_auc = evaluate_model(models, valX1,valY1, nclass_times, x_fold_type,fildername)	
		accuracys_final['train_loss1'].append(accuracys['train_loss1'])
		accuracys_final['train_acc1'].append(accuracys['train_acc1'])
		accuracys_final['val_loss2'].append(accuracys['val_loss2'])
		accuracys_final['val_acc2'].append(accuracys['val_acc2'])
		print_final_acc_loss(accuracys_final, nclass_times, x_fold_type,fildername)	
		print("pre_score, pre, rec, SN, SP, f1, mcc, roc_auc")
		print(pre_score, pre, rec, SN, SP, f1, mcc, roc_auc)		
		compile_flag = compile_flag + 1		
		pass
	
	
	



def run_model(nclass_times,compile_flag,one_of_ten_fold_file_path,fildername,models,train_neg,train_pos,val_neg,val_pos):
	path = one_of_ten_fold_file_path
	
	train_neg_data = np.load(path + train_neg[nclass_times])
	train_pos_data = np.load(path + train_pos[0])

	val_neg_data = np.load(path + val_neg[nclass_times])
	val_pos_data = np.load(path + val_pos[0])
	"""
	print(len(train_neg_data))
	print(len(train_pos_data))
	print(len(val_neg_data))
	print(len(val_pos_data))"""

	train_neg_data = pd.DataFrame(train_neg_data)
	train_pos_data = pd.DataFrame(train_pos_data)
	val_neg_data = pd.DataFrame(val_neg_data)
	val_pos_data = pd.DataFrame(val_pos_data)


	codingMode = 0

	train_all = pd.concat([train_pos_data, train_neg_data])
	trainX1, trainY1 = convertRawToXY(train_all.as_matrix(), codingMode=codingMode)		

	val_all = pd.concat([val_pos_data, val_neg_data])
	#print(val_all)
	valX1, valY1 = convertRawToXY(val_all.as_matrix(), codingMode=codingMode)
	#print(valX1)
	#print(valY1)

	models,accuracys = zyh_CNN(trainX1, trainY1, valX1, valY1, compiletimes=compile_flag, transferlayer=1,forkinas=1,compilemodels=models,fildername = fildername)
	
	return models,accuracys 


	


def print_final_acc_loss(accuracys_final, i, type,fildername):
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
	plt.legend(['train_loss1','val_loss2', 'train_acc1','val_acc2'], loc='upper left')
	plt.savefig('./'+fildername+'/' + str(type) + 'loss_acc%d.png' % i)
	
def using_ten_fold_crossing(ten_fold_file_path):
	path = ten_fold_file_path
	print(path)
	dirs = os.listdir(path)	
	compiletimes = 0
	for dir in dirs:
	    models = using_one_of_ten_fold_crossing(ten_fold_file_path + str(dir) + "/" , str(dir)+ "_","result")
	    #this place will save the model
	pass


if __name__ == "__main__":
	
	set_GPU(7)

	ten_fold_file_path = ".\\DTP_data\\ten_fold_data\\"
	#ten_fold_file_path = "./DTP_data/ten_fold_data/"
	
	using_ten_fold_crossing(ten_fold_file_path)