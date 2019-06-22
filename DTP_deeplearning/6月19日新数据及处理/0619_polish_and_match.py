import os
import json
import datetime
import pickle

space_counter = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def add_to_new_file_all(filename, name, pdb_proteins, site_proteins, site_proteins_new, site_numbers):
    "写成文件保存在外，输入的是文件名字，第一行内容，第二行内容，回车调整格式"
    outfopen = open(filename, 'a',encoding='UTF-8')
    outfopen.writelines(name)
    outfopen.writelines('\n')
    outfopen.writelines(pdb_proteins)
    outfopen.writelines('\n')
    outfopen.writelines(site_proteins)
    outfopen.writelines('\n')
    outfopen.writelines(site_proteins_new)
    outfopen.writelines('\n')
    outfopen.writelines(site_numbers)
    outfopen.writelines('\n')

    outfopen.close()
    pass

def parser_site_all_seq_into_dic_and_output(input_file,output_file):
    """解析文件字段，生成字典待用"""
    f = open(input_file, encoding='UTF-8')
    lines = f.readlines()
    seq_dic = {}
    seq_name = []
    filename = output_file
    for x in range(len(lines)):
        if (lines[x][0] == ">"):
            name = lines[x][0:7]
            pdb_proteins = lines[x + 1][0:-1]
            site_proteins = lines[x + 2][0:-1]
            site_numbers = lines[x + 3][0:-1]
            seq_dic[name] = [name,pdb_proteins,site_proteins,site_numbers]
            #print (seq_dic[name])
            name, pdb_proteins, site_proteins, site_proteins_new, site_numbers = polish_and_match(seq_dic[name],filename)
            add_to_new_file_all(filename, name, pdb_proteins, site_proteins, site_proteins_new, site_numbers)
    f.close()
    return seq_dic,seq_name

def  make_new_sequence(seq_dic):
    name = seq_dic[0]
    pdb_proteins = seq_dic[1]
    site_proteins = seq_dic[2]
    site_numbers =seq_dic[3]
    location = 0
    total_number = 0
    _number = 0
    site_proteins_new = site_proteins
    site_numbers_new = site_numbers
    del_space_number = 0

    """
    print(name)
    print(pdb_proteins)
    print(site_proteins)
    print(site_proteins_new)
    print(site_numbers)
    """
    for site_proteins_x in site_proteins_new:
        location = location + 1    #location is recode of site_protein's location
        #print(location)
        if site_proteins_x  == "_"  :
            total_number = total_number + 1    #mark a "_" and count the "_"'s number.
        elif site_proteins_x != "_" and total_number != 0:
            _location = location - total_number
            site_seq_string = site_proteins[_location - 3: _location + total_number + 2]
            #print(site_seq_string)
            if total_number >= 319:
                site_seq_string = site_proteins[_location - 3: _location]  + site_proteins[_location + 320: _location + total_number + 2]
                _number = total_number - 320
                #print(site_seq_string)
                #print(_number)
            else:
                _number = total_number

            for x in range(len(pdb_proteins)):

                if pdb_proteins[x - 2:x] == site_seq_string[:2]:
                    if total_number >= 320 and site_seq_string[ -3:] == pdb_proteins[ x + _number  : x + _number + 3]:
                        pass
                        # 7_0 del 320 space
                        fro = site_proteins_new[0:_location - 3]
                        mid = pdb_proteins[x - 2:x + _number + 3]
                        lst = site_proteins_new[_location + 320 + _number + 2:]
                        site_proteins_new = fro + mid + lst
                        space_counter[6] = space_counter[6] + 1
                        site_numbers_new = site_numbers_new[0:_location + _number - 1 ]  + site_numbers_new[_location + 320 + _number + 2 -3:]
                        break

                    elif pdb_proteins[ x + _number: x + _number + 3] == site_seq_string[ -3:]:
                        #1  AB_DEF(ABCDEF)  -> ABCDEF
                        pass
                        _location = _location - del_space_number
                        site_proteins_new = site_proteins_new[0:_location - 3] + pdb_proteins[
                                                                                 x - 2:x + _number + 3] + site_proteins_new[
                                                                                                          _location + _number + 2:]
                        space_counter[0] = space_counter[0] + 1
                        break

                    elif pdb_proteins[x: x + 3] == site_seq_string[ -3:]:
                        #2 AB__?__CDEF (ABCDEF) -> ABCDEF  #number changed
                        pass
                        _location = _location - del_space_number
                        fro = site_proteins_new[0:_location - 1]
                        mid = ""
                        lst = site_proteins_new[_location + total_number - 1:]
                        site_proteins_new = fro + lst
                        del_space_number = del_space_number + _number
                        space_counter[1] = space_counter[1] + 1
                        #change the numbers
                        site_numbers_new = site_numbers_new[0:_location - 1] + site_numbers_new[_location + total_number - 1:]

                        break

                    elif pdb_proteins[x + _number - 1: x + _number + 3 - 1] == site_seq_string[ -3:]:
                        pass
                        #3 fit ? letter and del 1 space
                        site_proteins_new = site_proteins_new[0:_location - 1] + pdb_proteins[x:x + _number + 3 + len(
                            site_proteins_new[_location + _number + 2:]) -1 ]
                        del_space_number = del_space_number + 1

                        space_counter[2] = space_counter[2] + 1
                        # 改变了长度！
                        site_numbers_new = site_numbers_new[0:_location - 1] + site_numbers_new[_location  :]

                        break

                    elif pdb_proteins[ x + _number + 3: x + _number + 3 + 3] == site_seq_string[-3:]:
                        #4 add 3 letter
                        site_proteins_new = site_proteins_new[0:_location - 3] + pdb_proteins[
                                                                                 x - 2:x + _number + 3 + 3] + site_proteins_new[
                                                                                                              _location + _number + 2:]
                        space_counter[3] = space_counter[3] + 1
                        break
                        # 改变了长度！
                        pass
                    elif pdb_proteins[ x + _number - 3: x + _number -3  + 3] == site_seq_string[-3:]:
                        #5 cut 3 space
                        fro = site_proteins_new[0:_location - 3]
                        mid = pdb_proteins[x - 2 - 3 : x + _number - 3]
                        lst = site_proteins_new[_location + _number + 2:]
                        site_proteins_new = fro + mid + lst
                        space_counter[4] = space_counter[4] + 1
                        break


                    elif pdb_proteins[x + _number + 1: x + _number + 3 + 1] == site_seq_string[-3:]:
                        pass
                        # 6 fit ? letter and add 1 letter #number changed
                        fro = site_proteins_new[0:_location - 1]
                        mid = pdb_proteins[x : x + _number + 1 ]
                        lst = site_proteins_new[_location + _number - 1:]
                        site_proteins_new = fro + mid + lst
                        space_counter[5] = space_counter[5] + 1
                        #change number, add 0

                        site_numbers_new = site_numbers_new[0:_location - 1] + site_numbers_new[_location - 1 : _location + _number - 1 ] + "0" + site_numbers_new[_location + _number - 1:]
                        break

                    elif total_number >= 320 and site_seq_string[ -3:] == pdb_proteins[ x + _number  : x + _number + 3]:
                        pass
                        """
                        # 7_0 del 320 space
                        #print(_number)
                        print(pdb_proteins[ x + _number  : x + _number + 3])
                        print(site_seq_string[ -3:])
                        print(pdb_proteins[x + _number: x + _number + 3])
                        print(site_seq_string[-3:])

                        fro = site_proteins_new[0:_location - 3]
                        mid = pdb_proteins[x - 2:x + _number + 3]
                        lst = site_proteins_new[_location + 320 + _number + 2:]
                        site_proteins_new = fro + mid + lst
                        space_counter[6] = space_counter[6] + 1
                        site_numbers_new = site_numbers_new[0:_location + _number - 1]  + site_numbers_new[_location + 320 + _number + 2:]
                        break"""


                    elif pdb_proteins[x + _number - 6 : x + _number - 6 + 3] == site_seq_string[ -3:]:
                        pass
                        #8 fit ? letter and del 6 space
                        fro = site_proteins_new[0:_location - 1]
                        mid = pdb_proteins[x: x + _number ]
                        lst = site_proteins_new[_location + _number - 1 + 6:]
                        site_proteins_new = fro + mid + lst

                        del_space_number = del_space_number + 6
                        # 改变了长度！
                        space_counter[7] = space_counter[7] + 1
                        site_numbers_new = site_numbers_new[0:_location - 1] + site_numbers_new[
                                                                               _location - 1: _location + _number - 1 -6]  + site_numbers_new[
                                                                                                                               _location + _number - 1:]

                        break
                    elif pdb_proteins[x + _number - 71 : x + _number - 71 + 3] == site_seq_string[ -3:]:
                        pass
                        #9 fit ? letter and del 71 space
                        fro = site_proteins_new[0:_location - 1]
                        mid = pdb_proteins[x: x + _number ]
                        lst = site_proteins_new[_location + _number - 1:]
                        site_proteins_new = fro + mid + lst
                        del_space_number = del_space_number + 71
                        # 改变了长度！
                        space_counter[8] = space_counter[8] + 1
                        break

                    elif "_" in site_proteins_new:
                        pass
                        #print(name)


                else:
                    pass
            total_number = 0    # delete “_”counter,it must be put at the end of the for loop

            #site_proteins_new = site_proteins_new  # mabey it is use less

            """print(name)
            print(pdb_proteins)
            print(site_proteins)
            print(site_proteins_new)
            print(site_numbers)"""



    return site_proteins_new,site_numbers_new


def polish_and_match(seq_dic,filename):
    name = seq_dic[0]
    pdb_proteins = seq_dic[1]
    site_proteins = seq_dic[2]
    site_numbers =seq_dic[3]
    site_proteins_new = site_proteins
    if "_" in site_proteins:
        site_proteins_new,site_numbers_new = make_new_sequence(seq_dic)
        #print(site_proteins)
        #print(site_proteins_new)
        if "_" in site_proteins_new:
            #print(name)
            filename = filename + "have_"
            add_to_new_file_all(filename, name, pdb_proteins, site_proteins, "", site_numbers)
            return "", "", "", "",""
        elif len(site_numbers_new)!= len(site_proteins_new):
            filename = filename + "have_"
            add_to_new_file_all(filename, name, pdb_proteins, site_proteins, "", site_numbers)
            return "", "", "", "",""
        return name, pdb_proteins, site_proteins, site_proteins_new, site_numbers_new
    else:
        return name, pdb_proteins, site_proteins, site_proteins_new, site_numbers


if __name__ == "__main__":
    #site_SEQ = "/home/zhaiyh884/1809project/PDB_find_seq/seq_from_pdb_all.fasta"
    site_SEQ = "C:/Users/zhaiy/Desktop/待整理文件夹20180927-/polish_1108/seq_from_pdb_test.fasta"
    site_SEQ = "C:/Users/zhaiy/Desktop/待整理文件夹20180927-/polish_1108/seq_from_pdb_all_1.fasta"

    input_file ="0619_id_seq_itom_lable.fasta"
    output_file = "0619_id_seq_itom_lable_no_.fasta"

    parser_site_all_seq_into_dic_and_output(input_file,output_file)
    print(space_counter)
