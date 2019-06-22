import pickle
import os


lines = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_Seq/Drug/bounddrugtargets.txt','r').readlines()
hetlist = []
for each in lines:
    hetlist.append(each.replace('\n',''))
print(hetlist)
print(len(hetlist))


picnumlabel = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_Seq/Drug/hetlist.pickle','wb')
pickle.dump(hetlist,picnumlabel) 
    