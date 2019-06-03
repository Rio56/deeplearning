from DTP_get_data import get_data

import datetime


#

def main():

    sequence_dic, lable_dic, training_data_id = get_data()



if __name__ == "__main__":
	starttime = datetime.datetime.now()
	main()
	endtime = datetime.datetime.now()
	print("totaltime = " + str(endtime - starttime))













