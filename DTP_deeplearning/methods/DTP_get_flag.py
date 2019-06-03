# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import pandas as pd

def get_flag(sequence_dic, lable_dic,training_data_id , windows = 49,empty_aa = '_' ):
	"""
	:argument
	:return:
	"""
	pos_sample = []
	neg_sample = []
	true_neg_sample = []
	seq_list_2d = []
	id_list = []
	pos_sample_number = 0
	neg_sample_number = 0

	# windows means all length, hfindows
	hf_windows = int(windows / 2)  # hf_windows : half windows
	
	#for protein_id in training_data_id:
	#	pass
	
	
	# id in the id list is useful, others are useless
	for protein_id in training_data_id:
		protein_id = protein_id.rstrip()
		seq = sequence_dic[protein_id]  # get the right seq
		if (protein_id in lable_dic):
			seq_lable = lable_dic[protein_id]
		else:
			seq_lable = []
		################################ form data in every sequerence ################################
		for pos in range(len(seq)):
			################################ each side of the seq ################################
			mid_aa = seq[pos]  # mid_aa : middle amino acid
			start = 0
			if pos - hf_windows > 0:
				start = pos - hf_windows
			left_seq = seq[start:pos]
			
			end = len(seq)
			if pos + hf_windows < end:
				end = pos + hf_windows + 1
			right_seq = seq[pos + 1:end]
			################################ make up the lack place ################################
			if len(left_seq) < hf_windows:
				if empty_aa is None:
					continue
				nb_lack = hf_windows - len(left_seq)
				left_seq = ''.join([empty_aa for _count in range(nb_lack)]) + left_seq
			
			if len(right_seq) < hf_windows:
				if empty_aa is None:
					continue
				nb_lack = hf_windows - len(right_seq)
				right_seq = right_seq + ''.join([empty_aa for _count in range(nb_lack)])
			################################ add the left mid right ################################
			final_seq = left_seq + mid_aa + right_seq
			
			if seq_lable[pos] == 1:
				pos_sample.append(final_seq)
				final_seq_list = [1] + [AA for AA in final_seq]
				pos_sample_number = pos_sample_number + 1
				id_list.append(protein_id)
				seq_list_2d.append(final_seq_list)
			elif seq_lable[pos] == 0:
				neg_sample.append(final_seq)
				final_seq_list = [0] + [AA for AA in final_seq]
				neg_sample_number = neg_sample_number + 1
				#id_list.append(protein_id)
				#seq_list_2d.append(final_seq_list)

				if 1 in seq_lable[pos - hf_windows:pos + hf_windows]:
					pass
				elif "_" in final_seq:
					pass
				# print(final_seq)
				else:
					id_list.append(protein_id)				
					seq_list_2d.append(final_seq_list)
				
				pass
			pass
		pass
	#seq_list_2d_df = pd.DataFrame(seq_list_2d)
	seq_list_2d_df = pd.DataFrame(seq_list_2d)

	return pos_sample, neg_sample, true_neg_sample, seq_list_2d_df

	

