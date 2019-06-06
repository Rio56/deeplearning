
from DTP_get_data import get_data,prepare_data
import os

file_path = ""
data_number = 1

sequence_dic, lable_dic, train_data_id, val_data_id,data_number= get_data(data_number,file_path)

#print(sequence_dic, lable_dic, train_data_id, val_data_id,data_number)

print(len(sequence_dic))
print(len(val_data_id))
val_data_id_set = set(val_data_id)
#print(val_data_id_set)
#print(len(val_data_id_set))
counter = 0

for key in sequence_dic:
    counter = counter + 1
    ID = key
    #print(counter)
    seq = sequence_dic[ID]
    #print(seq)
    
    if os.path.exists(".\\3232_lonly_fasta\\"+ str(ID) + ".fasta"):
        print(ID)
        
    file = open(".\\3232_lonly_fasta\\"+ str(ID) + ".fasta","w")
    file.write(">" + str(ID)  + "\n")
    file.write(str(seq))
    file.close()
    
    
    
    
