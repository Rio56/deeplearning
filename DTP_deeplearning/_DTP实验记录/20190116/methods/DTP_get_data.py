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
from sklearn import cross_validation
import pickle


def get_data():
	"""
	:argument
	:return:
	"""
	print("!")
	
	# different computer's path
	file_path = "D:\\drugtaget_prediction\\zyh_data\\"
	# file_path = "/home/zhaiyh/drug_target_gpu_test/zyh_data/"
	# file_path = "C:\\Users\\_Rio56\\Desktop\\drug_target_gpu_test\\zyh_data\\DTP_data\\"
	
	################################ set the file name ################################
	sequence_dic = file_path + "seqdicts.pickle"
	# '1MQD_A':'ASDFLHFSDHFKJLHWQERF...'
	
	lable_dic = file_path + "labeldicts.pickle"
	# '1MQD_A':[0,0,0,1,O,O,O,O,0....]
	
	# training_data = file_path + "seq_3cutoff.pickle"
	training_data = file_path + "seq_from_pdb_all_no_3232.pickle"
	#['5WC5_B', '2PX8_B', '4PXK_A'....]
	################################ open file and load pickle ################################
	sequence_dic = open(sequence_dic, 'rb')
	sequence_dic = pickle.load(sequence_dic)
	
	lable_dic = open(lable_dic, 'rb')
	lable_dic = pickle.load(lable_dic)
	
	training_data_id= open(training_data, 'rb')
	training_data_id = pickle.load(training_data_id)
	
	################################ return data ################################
	return sequence_dic, lable_dic, training_data_id


def prepare_data(data_with_flag):
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
	train_pos = trainX[np.where(trainX[:, 0] == 1)]
	train_neg = trainX[np.where(trainX[:, 0] != 1)]
	train_pos = pd.DataFrame(train_pos)
	train_neg = pd.DataFrame(train_neg)
	
	change_data_into_ten_fold(train_pos,train_neg)
	
	output_train_pos = open('train_pos.pkl', 'wb')
	pickle.dump(train_pos, output_train_pos)
	output_train_pos.close()
	
	output_train_neg = open('train_neg.pkl', 'wb')
	pickle.dump(train_neg, output_train_neg)
	output_train_neg.close()	

	return train_pos,train_neg

def change_data_into_ten_fold(train_pos,train_neg):
	#Residues = train_pos
	print("train_pos" + str(len(train_pos)))
	print("train_neg" + str(len(train_neg)))
	seed = 6
	np.random.seed(seed)
	kf_pos = cross_validation.KFold(len(train_pos), n_folds=10, shuffle=True,random_state=seed)
	kf_neg = cross_validation.KFold(len(train_neg), n_folds=10, shuffle=True,random_state=seed)
	print(kf_pos)
	print(kf_neg)
	i = 1
	times = int(len(train_neg)/len(train_pos))
	for train_index,test_index in kf_pos:
		#print(train_index,test_index)
		if i <= 10:
			x = train_index
			y = test_index
			save_Bootstrapping_data(train_pos,train_index,i,1,0.8,"train_pos_")
			save_Bootstrapping_data(train_pos,test_index,i,1,0.8,"test_pos_")
			i = i + 1
		#print(train_index)
		#print(test_index)
	print("pos_done~~~~~~~~~~~~~~")
		
	j = 1
	for train_index,test_index in kf_neg:
		print("kf_neg")
		if j <= 10:
			print("j = " + str(j))
			x = train_index
			y = test_index
			save_Bootstrapping_data(train_neg,train_index,j,times,0.8,"train_neg_")
			save_Bootstrapping_data(train_neg,test_index,j,times,0.8,"test_neg_")			
			j = j + 1
		#print(train_index)
		#print(test_index)	
	print("neg_done~~~~~~~~~~~~~~")
	

def save_Bootstrapping_data(data,tenfold_list,tenfload,times,srate = 0.8,tag = "tag"):
	#path = 'C:\\Users\\_Rio56\\Desktop\\drug_target_gpu_test\\zyh_DTP\\DTP_data\\ten_fold_data\\'

	path = 'D:\\drugtaget_prediction\\ten_fold_data\\'

	path = path + str(tenfload) + '\\'

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
			name = path + 'tenfold_' + str(tenfload) + '_' + tag +str(nclass_times) + '.npy'
			print(name)
			#print(counter)
			np.save(name,newdata)	
			nclass_times = nclass_times + 1
			newdata = pd.DataFrame(columns = data.columns.values.tolist())
			counter = 1			
		else:
			#print(counter)
			pass
			
			
		