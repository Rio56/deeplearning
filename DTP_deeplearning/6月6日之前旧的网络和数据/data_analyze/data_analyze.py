
from methods.DTP_get_data import get_data,prepare_data



import datetime


def analyze_one_data(lables):
	#length of protein
	protein_len = len(lables)
	print(protein_len)

	#number of positive sites
	pos_counter = 0
	for item in lables:
		if item == 1:
			pos_counter = pos_counter + 1

	print(pos_counter)

	return protein_len,pos_counter







	pass

def main():
	sequence_dic, lable_dic, training_data_id = get_data()
	protein_len_total = []
	pos_counter_total = []


	for protein_id in training_data_id:
		protein_id = protein_id.rstrip()
		seq_lable = lable_dic[protein_id]
		protein_len, pos_counter = analyze_one_data(seq_lable)

		protein_len_total.append(protein_len)
		pos_counter_total.append(pos_counter)
	print(protein_len_total)
	print(pos_counter_total)


#1数样本个数

    #数样本长度

    #数样本中1的个数

    #样本以5滑窗，其中有1的个数。






















if __name__ == "__main__":
	starttime = datetime.datetime.now()
	main()
	endtime = datetime.datetime.now()