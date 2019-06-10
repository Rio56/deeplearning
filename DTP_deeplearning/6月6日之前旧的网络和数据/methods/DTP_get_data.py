# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import numpy as np
import pandas as pd
#from sklearn import cross_validation
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import random

import pickle


def get_data(data_number,file_path):
	"""
	:argument
	:return:
	"""

	
	################################ set the file name ################################
	sequence_dic = file_path + "seqdicts.pickle"
	# '1MQD_A':'ASDFLHFSDHFKJLHWQERF...'
	
	lable_dic = file_path + "labeldicts.pickle"
	# '1MQD_A':[0,0,0,1,O,O,O,O,0....]

	training_data = file_path + "train.pickle"
	val_data = file_path + "val.pickle"

	#['5WC5_B', '2PX8_B', '4PXK_A'....]
	################################ open file and load pickle ################################
	sequence_dic = open(sequence_dic, 'rb')
	sequence_dic = pickle.load(sequence_dic)
	
	lable_dic = open(lable_dic, 'rb')
	lable_dic = pickle.load(lable_dic)
	
	training_data_id= open(training_data, 'rb')
	training_data_id = pickle.load(training_data_id)

	val_data_id = open(val_data, 'rb')
	val_data_id = pickle.load(val_data_id)
	
	################################ return data ################################
	print("1")
	#print(sequence_dic)
	print("2")
	#print(lable_dic)
	print("3")
	#print(training_data_id)
	print(len(training_data_id))
	print("4")
	return sequence_dic, lable_dic, training_data_id, val_data_id, data_number


def prepare_data(data_with_flag, data_number,windows,data_class):
	"""
	:argument
	:return:
	:ten_fold
	"""
	data_with_flag = data_with_flag.as_matrix()
	trainX = data_with_flag
	# print(trainfile)
	# get: train_pos  train_neg
	print(np.where(trainX[:, 0] == 1))
	pos = trainX[np.where(trainX[:, 0] == 1)]
	neg = trainX[np.where(trainX[:, 0] != 1)]
	pos = pd.DataFrame(pos)
	neg = pd.DataFrame(neg)



	"""for x in range(10):
		x_th_file = x + 1
		upset_data_into_ten_file(pos,neg,x_th_file,data_number,windows,data_class)"""
	x_th_file = 1

	upset_data_into_ten_file(pos,neg,x_th_file,data_number,windows,data_class)
	#change_data_into_ten_fold(pos,neg,data_number,windows)
	
	output_pos = open(data_class + '_pos.pkl', 'wb')
	pickle.dump(pos, output_pos)
	output_pos.close()
	
	output_neg = open(data_class + '_neg.pkl', 'wb')
	pickle.dump(neg, output_neg)
	output_neg.close()
	print("done")

	return pos,neg

def upset_data_into_ten_file(pos,neg,x_th_file,data_number,windows,data_class):
	print(data_class)


	list_pos = [i for i in range(len(pos))]
	list_neg = [i for i in range(len(neg))]

	random.shuffle(list_pos)
	random.shuffle(list_neg)

	times = int(len(neg) / len(pos))

	save_Bootstrapping_data(pos, list_pos, x_th_file, 1, data_number, windows, 0.8, data_class + "_pos_")
	save_Bootstrapping_data(neg, list_neg, x_th_file, times, data_number, windows, 0.8, data_class + "_neg_")

	pass

def change_data_into_ten_fold(train_pos,train_neg, data_number,windows):
	#Residues = train_pos
	print("train_pos" + str(len(train_pos)))
	print("train_neg" + str(len(train_neg)))
	seed = 6
	np.random.seed(seed)
	kf_pos = KFold(n_splits=10, shuffle=True,random_state=seed)
	kf_neg = KFold(n_splits=10, shuffle=True,random_state=seed)

	#kf_pos = cross_validation.KFold(len(train_pos), n_folds=10, shuffle=True, random_state=seed)
	#kf_neg = cross_validation.KFold(len(train_neg), n_folds=10, shuffle=True, random_state=seed)

	print(kf_pos)
	print(kf_neg)
	i = 1
	times = int(len(train_neg)/len(train_pos))
	for train_index,test_index in kf_pos.split(train_pos):
		print(train_index,test_index)
		if i <= 10:
			x = train_index
			y = test_index
			save_Bootstrapping_data(train_pos,train_index,i,1,data_number,windows,0.8,"train_pos_")
			save_Bootstrapping_data(train_pos,test_index,i,1,data_number,windows,0.8,"test_pos_")
			i = i + 1
		#print(train_index)
		#print(test_index)
	print("pos_done~~~~~~~~~~~~~~")
		
	j = 1
	for train_index,test_index in kf_neg.split(train_neg):
		print("kf_neg")
		if j <= 10:
			print("j = " + str(j))
			x = train_index
			y = test_index
			save_Bootstrapping_data(train_neg,train_index,j,times,data_number,windows,0.8,"train_neg_")
			save_Bootstrapping_data(train_neg,test_index,j,times,data_number,windows,0.8,"test_neg_")
			j = j + 1
		#print(train_index)
		#print(test_index)	
	print("neg_done~~~~~~~~~~~~~~")
	

def save_Bootstrapping_data(data,tenfold_list,ten_times,times, data_number,windows,srate = 0.8,tag = "tag"):

	path = './DTP_data_no_rdc/10times_train_' + data_number + '_lenth' + str(windows) + '/'
	path = path + str(ten_times) + '/'

	newdata =pd.DataFrame(columns = data.columns.values.tolist())
	counter = 1
	nclass_times =0
	slength = int(len(tenfold_list)/times)

	for item in tenfold_list:
		#print(data.loc[item].values )
		data_line = data.loc[item].values
		data_line = pd.DataFrame(data_line) 
		data_line = data_line.T
		newdata = pd.concat([newdata,data_line],ignore_index = True)
		#print(counter)
		counter = counter + 1
		if counter == slength :
			name = path + 'ten_times_' + str(ten_times) + '_' + tag +str(nclass_times) + '.npy'
			print(name)
			#print(counter)
			np.save(name,newdata)	
			nclass_times = nclass_times + 1
			newdata = pd.DataFrame(columns = data.columns.values.tolist())
			counter = 1			
		else:
			#print(counter)
			pass
			
			
		