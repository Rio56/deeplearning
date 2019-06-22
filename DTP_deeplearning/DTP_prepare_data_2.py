# -*- coding: utf-8 -*-
"""
# @Time    : 2018/6/7 13:06
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""
from methods_2.DTP2_data_cls import DTP_prepare_data
import datetime
import numpy as np
import pickle


def main(Dp):
    seq_dicts = "seq_dicts.pkl"
    lable_dicts = "lable_dicts.pkl"
    protein_id_all = "all_id_list.pkl"
    protein_id_03 = "seq_3cutoff.pickle"



    window = 31
    starttime = datetime.datetime.now()

    ########## Get Positive Samples ##########
    pos_seq_dicts, pos_lable_dicts, pos_protein_id = Dp.get_data(protein_id_all, seq_dicts, lable_dicts, "windows")
    # load data for pos, that is :protein_id_all

    pos_seq_one_hot_dicts = Dp.one_hot(pos_seq_dicts)
    pos_seq_phy_che_dicts = Dp.physico_chemical(pos_seq_dicts)
    # change data into one hot key \ physical_chemical\ pssm
    # pos_seq_feature_dicts = Dp.feature_fusion(pos_seq_one_hot_dicts, pos_seq_physico_chemical_dicts)
    # this methods can add two feature into one matrix

    pos_one_hot_set = Dp.divide_pos_neg(pos_seq_one_hot_dicts, pos_lable_dicts, pos_protein_id, window, set={},
                                        tag="pos_one_hot")
    pos_phy_che_set = Dp.divide_pos_neg(pos_seq_phy_che_dicts, pos_lable_dicts, pos_protein_id, window, set={},
                                        tag="pos_phy_che")
    # find the positive samples

    Dp.divide_sets(pos_one_hot_set, tag="pos_one_hot", train=8, val=1, test=1, system="windows")
    Dp.divide_sets(pos_phy_che_set, tag="pos_phy_che", train=8, val=1, test=1, system="windows")
    # save the items with proper name

    endtime = datetime.datetime.now()
    print("positive_done = " + str(endtime - starttime))

    ########## Get Negtive Samples ##########
    neg_seq_dicts, neg_lable_dicts, neg_protein_id = Dp.get_data(protein_id_all, seq_dicts, lable_dicts, "windows")
    # load data for pos, that is :protein_id_03 use the 0.3 cd hit cut off to eliminate the redundancy
    neg_seq_one_hot_dicts = Dp.one_hot(neg_seq_dicts)
    neg_seq_physico_chemical_dicts = Dp.physico_chemical(neg_seq_dicts)
    # change data into one hot key \ physical_chemical\ pssm
    # neg_seq_feature_dicts = Dp.feature_fusion(neg_seq_one_hot_dicts, neg_seq_physico_chemical_dicts)
    # this methods can add two feature into one matrix

    neg_one_hot_set = Dp.divide_pos_neg(neg_seq_one_hot_dicts, neg_lable_dicts, neg_protein_id, window, set={},
                                        tag="neg_one_hot")
    neg_phy_che_set = Dp.divide_pos_neg(neg_seq_physico_chemical_dicts, neg_lable_dicts, neg_protein_id, window, set={},
                                        tag="neg_phy_che")
    # find the negative samples
    Dp.divide_sets(neg_one_hot_set, tag="neg_one_hot", train=8, val=1, test=1, system="windows")
    Dp.divide_sets(neg_phy_che_set, tag="neg_phy_che", train=8, val=1, test=1, system="windows")
    endtime = datetime.datetime.now()
    print("negative_done = " + str(endtime - starttime))


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    Data_prepare = DTP_prepare_data()
    main(Data_prepare)
    endtime = datetime.datetime.now()
    print("totaltime = " + str(endtime - starttime))
