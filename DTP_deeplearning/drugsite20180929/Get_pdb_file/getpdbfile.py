import pickle
import os
import shutil


file = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_IDmapping/gongIDmapping.pickle','rb')
unplist = pickle.load(file)

pdbfile  = []
for each in unplist:
    
    for eachitem in each['PDBID']:
        path = "/home/BiodataTest/updb/pdb/" + eachitem[1:3].lower() +"/pdb"+ eachitem.lower() +".ent"
        tap = False
        try:           
            lines = open(path,'r').readlines()
            for line in lines :
                if line.split()[0] == "SITE":
                    tap = True            
        except:
            print(path)


        if tap == True:
            pdbfile.append(path)  
            try:
                os.makedirs("/home/RaidDisk/gongjt057/drugsite20180929/Get_PDB_file/PDB_file/SITE/" + each['UniProtID'] )
            except:
                pass
            dst = "/home/RaidDisk/gongjt057/drugsite20180929/Get_PDB_file/PDB_file/SITE/" + each['UniProtID'] +"/pdb"+ eachitem.lower() +".ent"
            shutil.copy(path, dst)

print(len(set(pdbfile)))
print(len(pdbfile))

#HETfile 10531/13371

                    
