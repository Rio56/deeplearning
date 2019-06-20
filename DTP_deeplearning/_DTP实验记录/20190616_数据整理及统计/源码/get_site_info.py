from PDBParseBase import PDBParserBase    #get_site_header_seq_info
import time, os,datetime,logging,gzip,pickle    #get_site_header_seq_info


def mkdir(path):
    #Created uncompress path folder
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + " Created folder sucessful!")
        return True
    else:  
        #print ("this path is exist")
        return False   


def get_site_header_seq_info(rootdir,savefilepath):
    """extract header\sequence\site\remark800 info in rootdir.
    and then, save them as a pickle content with list[1_site,2_header,3_sequence]
    
    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0615_data"
    
    scan all pdb files need about 60 min.
    
    """

    count = 0
    counter_mem = 0
    pdbbase = PDBParserBase()
    """
    #test cetern item
    pdb_header_info = pdbbase.get_header_info('/home/RaidDisk/pdbfiles/updb/pdb/a2/pdb2a2q.ent')
    pdb_site_info = pdbbase.get_site_info('/home/RaidDisk/pdbfiles/updb/pdb/a2/pdb2a2q.ent')
    pdb_seq_info = pdbbase.get_sequence_fromSEQ('/home/RaidDisk/pdbfiles/updb/pdb/a2/pdb2a2q.ent')
    print(pdb_header_info)
    print(pdb_site_info)
    print(pdb_seq_info)   """

    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            data_3_items = []
            #analyzedata
            pdb_site_info = pdbbase.get_site_info(os.path.join(parent,filename))
            data_3_items.append(pdb_site_info)          
            if not pdb_site_info :
                print("do not have site:" + filename)
                "in order to save some time"
                continue
            
            pdb_header_info = pdbbase.get_header_info(os.path.join(parent,filename))
            data_3_items.append(pdb_header_info)
            #print(pdb_header_info)      
            
            pdb_seq_info = pdbbase.get_sequence_fromSEQ(os.path.join(parent,filename))
            data_3_items.append(pdb_seq_info)
            #print(pdb_seq_info)               
            
            #save data
            if not pdb_site_info :
                pass                                         
            else:
                dirname = filename[4:6]       
                new_Filepath = savefilepath +"/" +  str(dirname)+"/"
                mkdir(new_Filepath)  
                new_filename =  filename[3:7] + ".pickle"      
                
                with open(new_Filepath + new_filename,"wb") as dbFile:
                    pickle.dump(data_3_items,dbFile)         
                """with open(new_Filepath + new_filename,"rb") as dbFile:
                    file = pickle.load(dbFile)  """
                pass
    pass

if __name__ == "__main__": 

    start = datetime.datetime.now()

    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0615_data"
    get_site_header_seq_info(rootdir,savefilepath)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)     














































