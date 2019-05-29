# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
from methods.DTP_set_GPU import set_GPU
from methods.DTP_get_data import get_data,prepare_data
from methods.DTP_get_flag import get_flag

import datetime

def main():
	#
	
	#
	sequence_dic, lable_dic, training_data_id = get_data()
	#
	pos_sample, neg_sample, true_neg_sample, data_with_flag = get_flag(sequence_dic, lable_dic, training_data_id)
	#
	pos_data, neg_data, = prepare_data(data_with_flag)
	#
	#start_training(train_pos_data, train_neg_data, val_pos_data, val_neg_data)
	pass

if __name__ == "__main__":
	starttime = datetime.datetime.now()
	main()
	endtime = datetime.datetime.now()
	print("totaltime = " + str(endtime - starttime))





