# -*- coding: utf-8 -*-
"""
# @Time    : 2018/12/20 13:45
# @Author  : zyh
# @Email   : 2536593870@qq.com
# @File    :
#@Others   :
"""

from DTP_get_data import get_data, prepare_data
from DTP_get_flag import get_flag
import pickle

import datetime


def main():
    #
    data_number = "1"
    "1,   2,   3"
    windows = 31
    "5   21   31   41"
    print("windows" + str(windows))
    # file_path = ".\\DTP_data_no_rdc\\"
    file_path = ""

    #
    sequence_dic, lable_dic, val_data_id, data_number = get_data(data_number, file_path)
    #

    pos_sample, neg_sample, true_neg_sample, val_data_with_flag = get_flag(sequence_dic, lable_dic, val_data_id,
                                                                           windows)
    # pos_data, neg_data = prepare_data(val_data_with_flag, data_number, windows, "val")

    # print(val_data_with_flag)

    pos = []
    neg = []

    for item in val_data_with_flag:
        if item[0] == "0":
            neg.append(item[1:])
        else:
            pos.append(item[1:])
    print(len(neg))
    print(len(pos))

    print(len(set(neg)))
    print(len(set(pos)))
    counter = 0

    output = open('data_old_neg_138822.pkl', 'wb')
    pickle.dump(neg, output)

    total = neg + pos

    print(len(set(total)))

    print(str(len(set(neg)) + len(set(pos)) - len(set(total))))


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    main()
    endtime = datetime.datetime.now()
    print("totaltime = " + str(endtime - starttime))
