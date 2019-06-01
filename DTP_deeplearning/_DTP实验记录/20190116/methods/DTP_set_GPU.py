# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import os
import tensorflow as tf

def set_GPU(gpu_id = 3):
	"""
	:argument
	:return:
	
	"""
	print(gpu_id)
	gpu_id = str(gpu_id)
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	os.system('echo $CUDA_VISIBLE_DEVICES')
	# zhiding gpu
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	tf.Session(config=tf_config)
	pass