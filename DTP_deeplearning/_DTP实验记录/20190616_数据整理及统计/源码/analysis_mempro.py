#/home/RaidDisk/pdbfiles/updb
from PDBParseBase import PDBParserBase

import time, os,datetime,logging,gzip,pickle


"""
def find_memberain_protein(rootdir,savefilepath):

    count = 0
    counter_mem = 0
    pdbbase = PDBParserBase()
    pdb_header_info = {}
    
  
    
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            #print(filename)
            count = count + 1
            dirname = filename[3:7]
            #print(dirname)
 
            pdb_header_info = pdbbase.get_header_info(os.path.join(parent,filename))
            #print(pdb_header_info)
            if "MEM" in pdb_header_info["HEADER_classification"]:
                counter_mem = counter_mem + 1
                print(counter_mem)
                print(pdb_header_info)
                cmd = 'cp ' + str(os.path.join(parent,filename)) + ' ' + str(os.path.join(savefilepath,filename)) 
                os.system(cmd)         
    pass"""

def analysis_membrane_protein(rootdir):
    count = 0
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            count = count + 1
            print(count)
            print(filename)
            dirname = filename[3:7]
            print(dirname)
            pdbbase = PDBParserBase()
            pdb_header_info = pdbbase.get_header_info(os.path.join(parent,filename))
            pdb_site_info = pdbbase.get_site_info(self, PDBfile)
            print(pdb_site_info)    
    pass




if __name__ == "__main__": 
    
    
    start = datetime.datetime.now()

    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/membrane_protein"
        
    #find_memberain_protein(rootdir,savefilepath)
    
    analysis_membrane_protein(savefilepath)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)     














































