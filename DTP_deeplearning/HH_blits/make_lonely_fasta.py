
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

#print(sequence_dic["5E7C_A"])
#print(sequence_dic["5E7C_a"])


for key in sequence_dic:

    seq = sequence_dic[key]
    #print(seq)
    
    if os.path.exists(".\\3232_lonly_fasta\\"+ str(key) + ".fasta"):
        print(key)
        counter = counter + 1
        #print(counter)
        
    file = open(".\\3232_lonly_fasta\\"+ str(key) + ".fasta","w")
    file.write(">" + str(key)  + "\n" + str(seq))
    file.close()
    
    
    
    
