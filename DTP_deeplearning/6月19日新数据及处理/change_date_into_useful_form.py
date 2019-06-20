import pickle
import time, os,datetime

def change_data_into_name_seq_lable(filename):

    f = open(filename, encoding='UTF-8')
    lines = f.readlines()
    id_list = []
    seq_dict = {}
    lable_dict = {}
    for x in range(len(lines)):
        if (lines[x][0] == ">"):
            name = lines[x][1:7]
            pdb_proteins = lines[x + 1][0:-1]
            site_proteins = lines[x + 2][0:-1]
            new_site_proteins = lines[x + 3][0:-1]
            site_numbers = lines[x + 4][0:-1]

        id_list.append(name)
        seq_dict[name] = new_site_proteins
        lable_dict[name] = [item for item in site_numbers]

    id_list_file = open('all_id_list.pkl', 'wb')
    pickle.dump(id_list, id_list_file)

    seq_dict_file = open('seq_dicts.pkl', 'wb')
    pickle.dump(seq_dict, seq_dict_file)

    lable_dictt_file = open('lable_dicts.pkl', 'wb')
    pickle.dump(lable_dict, lable_dictt_file)
    f.close()

    pass

if __name__ == "__main__":
    start = datetime.datetime.now()
    filename = "0619_id_seq_itom_lable_no_.fasta"
    change_data_into_name_seq_lable(filename)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end - start)

