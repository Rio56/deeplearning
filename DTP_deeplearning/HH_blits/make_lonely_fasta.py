
from DTP_get_data import get_data,prepare_data

file_path = ""
data_number = 1

sequence_dic, lable_dic, train_data_id, val_data_id,data_number= get_data(data_number,file_path)

#print(sequence_dic, lable_dic, train_data_id, val_data_id,data_number)

#print(sequence_dic)

for item in val_data_id:
    item = item.rsplit()
    print(item)
    seq = sequence_dic[item[0]]
    print(seq)
    file = open(""+ str(item[0]) + ".fasta","w")
    file.write(">" + str(item[0])  + "\n")
    file.write(str(seq))
    file.close()
