#/home/RaidDisk/pdbfiles/updb
from PDBParseBase import PDBParserBase

import time, os,datetime,logging,gzip,configparser,pickle



def find_memberain_protein(rootdir,savefilepath):
    #find all protein that header have "mem"in it and save them into savefilepath
    count = 0
    counter_mem = 0
    pdbbase = PDBParserBase()
    pdb_header_info = {}
    
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            count = count + 1
            dirname = filename[3:7]
 
            pdb_header_info = pdbbase.get_header_info(os.path.join(parent,filename))
            if "MEM" in pdb_header_info["HEADER_classification"]:
                counter_mem = counter_mem + 1
                cmd = 'cp ' + str(os.path.join(parent,filename)) + ' ' + str(os.path.join(savefilepath,filename)) 
                os.system(cmd)         
    pass

if __name__ == "__main__": 
    
    
    start = datetime.datetime.now()

    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/membrane_protein"
        
    find_memberain_protein(rootdir,savefilepath)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)     














































