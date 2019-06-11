# -*- coding: utf-8 -*-
"""
# @Time    : 2018/6/7 13:06
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
import pickle
import random
import numpy as np
import pandas as pd
import datetime
import keras.utils.np_utils as kutils


class DTP_prepare_data(object):

    def get_data(self, protein_id, seq_dicts="seq_dicts", lable_dicts="lable_dicts", system="linux"):
        """{id: seq[], ...}
            Enter the name of the file to be processed, have the path judgment and file selection judgment,
            read the list of protein sequences, the list of protein labels,
            the list of protein ids, and return the dictionary to be processed.
        """
        if system == "windows":
            filepath = ".\\DTP_data_2\\"
        else:
            filepath = "./DTP_data_2/"

        seq_dicts = open(filepath + seq_dicts, 'rb')
        seq_dicts = pickle.load(seq_dicts)

        lable_dicts = open(filepath + lable_dicts, 'rb')
        lable_dicts = pickle.load(lable_dicts)

        protein_id = open(filepath + protein_id, 'rb')
        protein_id = pickle.load(protein_id)

        return seq_dicts, lable_dicts, protein_id

    def one_hot(self, data_dicts):
        """Pass in the protein dictionary, encode the protein sequence 01, and pass it back.
        It's like changing the value of a dictionary key pair from a protein sequence to a protein sequence + one hot,
        （蛋白质序列可以直接转变编码，但其他特征是根据蛋白质序列算的，所以需要整个蛋白的信息，因此这里one_hot也针对蛋白质变换）
        seq_dicts_feature = {'protein_id': [[AA list], [AA one_hot list]], ...}"""

        dicts = {'C': '10000000000000000000', 'D': '01000000000000000000', 'S': '00100000000000000000',
                 'Q': '00010000000000000000', 'K': '00001000000000000000', 'I': '00000100000000000000',
                 'P': '00000010000000000000', 'T': '00000001000000000000', 'F': '00000000100000000000',
                 'N': '00000000010000000000', 'G': '00000000001000000000', 'H': '00000000000100000000',
                 'L': '00000000000010000000', 'R': '00000000000001000000', 'W': '00000000000000100000',
                 'A': '00000000000000010000', 'V': '00000000000000001000', 'E': '00000000000000000100',
                 'Y': '00000000000000000010', 'M': '00000000000000000001', 'X': '00000000000000000000'}
        seq_onehot = []
        item_list = []
        seq_dicts_feature = {}
        for item in data_dicts:
            item_list.append(item)
            seq = data_dicts[item]
            for AA in seq:
                if AA in seq:
                    aa_hot = dicts[AA]
                    seq_onehot.append([int(x) for x in list(aa_hot)])
                else:
                    print(AA)
                    aa_hot = dicts["X"]
                    seq_onehot.append([int(x) for x in list(aa_hot)])
            feature = []
            feature.append(list(data_dicts[item]))
            feature.append(seq_onehot)
            seq_dicts_feature[item] = feature
            seq_onehot = []
            item_list = []
        """for item in data_dicts:
            #show the data
            print(item)
            print(data_dicts[item])
            print(len(data_dicts[item][0]))
            print(np.array(data_dicts[item]).shape)"""
        return seq_dicts_feature

    def physico_chemical(self, data_dicts):
        """Pass in the protein feature dictionary, add physico_chemical in to its feature.
        It's like changing the value of a dictionary key pair from a protein sequence to a protein sequence + one hot,
        （蛋白质序列可以直接转变编码，但其他特征是根据蛋白质序列算的，所以需要整个蛋白的信息，因此这里one_hot也针对蛋白质变换）
        seq_dicts_feature = {'protein_id':[[ AA list],[ AA physico_chemical list]],...      }
        """
        dicts = {'A': [0.62, -0.5, 15, 2.35, 9.87, 6.11, 91.5, 89.09, 27.5, -0.06],
                 'C': [0.2900, -1.0000, 47.0000, 1.7100, 10.7800, 5.0200, 117.7, 121.15, 44.6, 1.36],
                 'D': [-0.9000, 3.0000, 59.0000, 1.8800, 9.6000, 2.9800, 124.5, 133.1, 40, -0.8],
                 'E': [-0.7400, 3.0000, 73.0000, 2.1900, 9.6700, 3.0800, 155.1, 147.13, 62, -0.77],
                 'F': [1.1900, -2.5000, 91.0000, 2.5800, 9.2400, 5.9100, 203.4, 165.19, 115.5, 1.27],
                 'G': [0.4800, 0, 1.0000, 2.3400, 9.6000, 6.0600, 66.4, 75.07, 0, -0.41],
                 'H': [-0.4000, -0.5000, 82.0000, 1.7800, 8.9700, 7.6400, 167.3, 155.16, 79, 0.49],
                 'I': [1.3800, -1.8000, 57.0000, 2.3200, 9.7600, 6.0400, 168.8, 131.17, 93.5, 1.31],
                 'K': [-1.5000, 3.0000, 73.0000, 2.2000, 8.9000, 9.4700, 171.3, 146.19, 100, -1.18],
                 'L': [1.0600, -1.8000, 57.0000, 2.3600, 9.6000, 6.0400, 167.9, 131.17, 93.5, 1.21],
                 'M': [0.6400, -1.3000, 75.0000, 2.2800, 9.2100, 5.7400, 170.8, 149.21, 94.1, 1.27],
                 'N': [-0.7800, 0.2000, 58.0000, 2.1800, 9.0900, 10.7600, 135.2, 132.12, 58.7, -0.48],
                 'P': [0.1200, 0, 42.0000, 1.9900, 10.6000, 6.3000, 129.3, 115.13, 41.9, 0],
                 'Q': [-0.8500, 0.2000, 72.0000, 2.1700, 9.1300, 5.6500, 161.1, 146.15, 80.7, -0.73],
                 'R': [-2.5300, 3.0000, 101.0000, 2.1800, 9.0900, 10.7600, 202, 174.2, 105, -0.84],
                 'S': [-0.1800, 0.3000, 31.0000, 2.2100, 9.1500, 5.6800, 99.1, 105.09, 29.3, -0.5],
                 'T': [-0.0500, -0.4000, 45.0000, 2.1500, 9.1200, 5.6000, 122.1, 119.12, 51.3, -0.27],
                 'V': [1.0800, -1.5000, 43.0000, 2.2900, 9.7400, 6.0200, 141.7, 117.15, 71.5, 1.09],
                 'W': [0.8100, -3.4000, 130.0000, 2.3800, 9.3900, 5.8800, 237.6, 204.24, 145.5, 0.88],
                 'Y': [0.2600, -2.3000, 107.0000, 2.2000, 9.1100, 5.6300, 203.6, 181.19, 117.3, 0.33],
                 'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

        seq_physico_chemical = []
        item_list = []
        seq_dicts_feature = {}
        for item in data_dicts:
            item_list.append(item)  # add protein name into list
            seq = data_dicts[item]  # find certein seq in the dic
            for AA in seq:
                if AA in seq:
                    aa_p_c = dicts[AA]
                    seq_physico_chemical.append(aa_p_c)
                else:
                    aa_p_c = dicts["X"]
                    seq_physico_chemical.append(aa_p_c)
            feature = []
            feature.append(list(data_dicts[item]))
            feature.append(seq_physico_chemical)
            seq_dicts_feature[item] = feature
            seq_physico_chemical = []
            item_list = []
        """for item in data_dicts:
            #show the data
            print(item)
            print(data_dicts[item])
            print(len(data_dicts[item][0]))
            print(np.array(data_dicts[item]).shape)"""
        return seq_dicts_feature

    def feature_fusion(self, feature_dic_1, feature_dic_2):
        """
        :param feature_dic_1: seq_dicts_feature = {'protein_id':[[ AA list],[ AA ????_1 list]],...      }
        :param feature_dic_2: seq_dicts_feature = {'protein_id':[[ AA list],[ AA ????_2 list]],...      }
        :return:
        """
        mixed_feature = {}
        feature_list = []
        for item in feature_dic_1:
            feature_list.append(feature_dic_1[item][0])  # add the seq (in list form)
            feature_1_np = np.array(feature_dic_1[item][1])
            feature_2_np = np.array(feature_dic_2[item][1])
            add_up = np.concatenate((feature_1_np, feature_2_np), axis=1)  # change into np form in order to add quickly
            feature_list.append(add_up.tolist())  # add the added features
            mixed_feature[item] = feature_list
            feature_list = []
        return mixed_feature

    def pssm(data_dicts):
        """{id: seq, ...} 对蛋白质添加pssm矩阵特征。（或单独提取）（后续步骤）, {id: [seq, 1, 2, ...,???]"""
        data_dicts_onehot = {}
        return data_dicts_onehot

    def divide_pos_neg(self, seq_feature_dicts, lable_dicts, protein_id, window, set, tag):
        """    {seque: filture[]}
    Read the lable label of the data to divide the positive and negative samples.
    Positive samples are eliminated from negative samples, and labels and protein data are docked.
    Complement processing is added to the deficient protein sequences.
    (divide positive sample first, divide negative sample later)
    """
        if "pos" in tag:
            set = {}
            flag = "1"
        else:
            flag = "0"

        make_up = []

        if "one_hot" in tag:
            #print("one_hot")
            make_up = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif "phy_che" in tag:
            #print("phy_che")
            make_up = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # decide positive sample and negative
        hf_window = int(window / 2)
        newset = {}
        counter = 0

        for item in protein_id:
            item = item.rstrip()
            if len(lable_dicts[item]) != len(seq_feature_dicts[item][0]) or len(lable_dicts[item]) != len(
                    seq_feature_dicts[item][1]):
                print(item)
                print(len(lable_dicts[item]))
                print(len(seq_feature_dicts[item]))
                # find defferent length item
            seq = ""
            feature = []
            for position in range(len(lable_dicts[item])):
                if position - hf_window >= 0 and position + hf_window < len(lable_dicts[item]):
                    start = position - hf_window
                    end = position + hf_window + 1
                    for aa in seq_feature_dicts[item][0][start:end]:
                        seq = seq + aa
                    for ftr in seq_feature_dicts[item][1][start:end]:
                        feature.append(ftr)
                    if len(seq) != window:
                        print(len(seq))
                        counter = counter + 1
                    # if the window is in the seq.
                elif position - hf_window < 0:
                    left_size = hf_window - position
                    end = window - left_size
                    for num in range(left_size):
                        seq = seq + "X"
                    for aa in seq_feature_dicts[item][0][0:end]:
                        seq = seq + aa
                    for num in range(left_size):
                        feature.append(make_up)
                    for ftr in seq_feature_dicts[item][1][0:end]:
                        feature.append(ftr)
                    if len(seq) != window:
                        print(seq)
                    # if the window is at the left side,we need add the "X" at first

                elif position + hf_window >= len(lable_dicts[item]):
                    right_size = position + hf_window - len(lable_dicts[item]) + 1
                    start = position - hf_window
                    for aa in seq_feature_dicts[item][0][start:]:
                        seq = seq + aa
                    for num in range(right_size):
                        seq = seq + "X"
                    for ftr in seq_feature_dicts[item][1][start:]:
                        feature.append(ftr)
                    for num in range(right_size):
                        feature.append(make_up)
                    if len(seq) != window:
                        print(seq)
                    # if the window is at the right side,we need add the "X" at last
                else:
                    print("!!!")

                if len(seq) != window:
                    print(seq)
                # if str(lable_dicts[item][position]) == flag:
                # if do not want to del the same seq_windows, we can use the upper one
                if str(lable_dicts[item][position]) == flag and seq not in set:
                    newset[seq] = feature
                seq = ""
                feature = []
                pass

        print(tag + str(len(newset.keys())))
        #test_file = open(tag + '_seq_data.txt', 'w')
        #test_file.write(str(newset.keys()))
        #test_file.close()
        return newset

    def divide_sets(self, data, tag="tag", train=8, val=1, test=1, system="windows"):
        """After random scrambling, the samples are divided into a certain proportion of training set,
        verification set and test set.The input sequence and its corresponding characteristics return these three
        types of data.After shuffling, choose in turn.Separate positive and negative samples.Save 6 documents in total.
        Save three copies at a time.Mark the positive and negative samples here.
        train_pos, val_pos, test_pos.
        train_neg, val_neg, test_neg."""
        if system == "windows":
            filepath = ".\\DTP_data_2\\"
        else:
            filepath = "./DTP_data_2/"

        data_list = list(data.values())
        random.shuffle(data_list)  # shuffle!!!!
        numbers = len(data_list)
        total_rate = train + val + test
        # print(numbers)

        train_set = data_list[0:int(train * numbers / total_rate)]
        val_set = data_list[int(train * numbers / total_rate):int((train + val) * numbers / total_rate)]
        test_set = data_list[int((train + val) * numbers / total_rate):]



        train_file = open(filepath + "train_" + tag + '_data.pkl', 'wb')
        pickle.dump(train_set, train_file)
        print( "train_" + tag + '_data : ' + str(len(train_set)))

        val_file = open(filepath + "val_" + tag + '_data.pkl', 'wb')
        pickle.dump(val_set, val_file)
        print("val_" + tag + '_data : ' + str(len(val_set)))

        test_file = open(filepath + "test_" + tag + '_data.pkl', 'wb')
        pickle.dump(test_set, test_file)
        print("test_" + tag + '_data : ' + str(len(test_set)))

        return

    def load_data(self, usage,feature, n_to_p_rate=1, random_rate=0.8, system="windows"):
        """读取过程中，区分正负样本。输入需要取的类别，返回可输入网络的文件。"train" - - train，val
    选相同的数值随机抽取（上载数据前计算train的0
    .8）。“val” 选取val组别中的正负样本。
    注意，针对行和列的问题，注意，标签处理问题，目标是将数据转换成可放入网络的格式：list
    appand后data
    fram.
"""
        if system == "windows":
            filepath = ".\\DTP_data_2\\"
        else:
            filepath = "./DTP_data_2/"

        if usage == "train":
            pos_data = filepath + "train_pos_" + feature + "_data.pkl"
            neg_data = filepath + "train_pos_" + feature + "_data.pkl"
        elif usage == "test":
            pos_data = filepath + "test_pos_" + feature + "_data.pkl"
            neg_data = filepath + "test_neg_" + feature + "_data.pkl"
        else:
            pos_data = filepath + "val_pos_" + feature + "_data.pkl"
            neg_data = filepath + "val_neg_" + feature + "_data.pkl"



        pos_data = open(pos_data, 'rb')
        neg_data = open(neg_data, 'rb')
        pos_data = pickle.load(pos_data)
        neg_data = pickle.load(neg_data)

        pos_lenth = len(pos_data)
        neg_lenth = len(neg_data)
        pos_lenth = pos_lenth * random_rate  # do not use all the pos_set in every training
        neg_lenth = pos_lenth * n_to_p_rate  # may load n_to_p_rate time

        random.shuffle(pos_data)
        random.shuffle(neg_data)

        pos_data = pos_data[0:int(pos_lenth)]
        neg_data = neg_data[0:int(neg_lenth)]

        pos_lenth = len(pos_data)
        neg_lenth = len(neg_data)
        data = pos_data + neg_data

        lable = [1 for i in range(pos_lenth)] + [0 for i in range(neg_lenth)]
        data_matrix = np.zeros((len(data), len(data[0]), len(data[0][0])))
        data_matrix_right_form = np.zeros((len(data), len(data[0][0]), len(data[0])))
        sample_number = 0
        AA_number = 0
        for seq_item in data:
            for AA_item in seq_item:
                #print(AA_item)
                data_matrix[sample_number][AA_number] = np.array(AA_item)
                AA_number += 1
            data_matrix_right_form[sample_number] = data_matrix[sample_number].T
            #change to the form that have meaning for cnn
            AA_number = 0
            sample_number += 1
        
        lable = kutils.to_categorical(lable)
        
        return data_matrix_right_form, lable

    def change_AA_into_onehot(self):
        dict = {'C': 0, 'D': 1, 'S': 2, 'Q': 3, 'K': 4,
                'I': 5, 'P': 6, 'T': 7, 'F': 8, 'N': 9,
                'G': 10, 'H': 11, 'L': 12, 'R': 13, 'W': 14,
                'A': 15, 'V': 16, 'E': 17, 'Y': 18, 'M': 19}
        print("x")
        new_dict = {}

        for item in dict:
            # print(item)
            # print(dict[item])
            num = dict[item]
            hotkey = ""
            for i in range(20):
                if i == num:
                    hotkey = hotkey + "1"
                else:
                    hotkey = hotkey + "0"
            print(hotkey)
            new_dict[item] = hotkey
        print(new_dict)

        pass


if __name__ == '__main__':
    # DTP_prepare_data().change_AA_into_onehot()

    starttime = datetime.datetime.now()
    Dp = DTP_prepare_data()
    Dp.load_data("val", n_to_p_rate=1, random_rate=0.8, system="windows")
    endtime = datetime.datetime.now()
    print("totaltime = " + str(endtime - starttime))

    pass
