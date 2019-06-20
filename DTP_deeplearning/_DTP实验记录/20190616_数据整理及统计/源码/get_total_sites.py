#/home/RaidDisk/pdbfiles/updb
from PDBParseBase import PDBParserBase

import time, os,datetime,logging,gzip,pickle

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


def get_site_info(rootdir,savefilepath):

    count = 0
    counter_mem = 0
    pdbbase = PDBParserBase()
    pdb_header_info = {}
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
                print("do not have site" + filename)
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

def find_all_sites(rootdir):

    total_site = []
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:
            # walk into one file
            #print(filename)
            pro_id = filename[0:4]
            file_path = os.path.join(parent, filename)
            with open(file_path, "rb") as dbFile:
                file = pickle.load(dbFile)
            #print(file)
            site = file[0]
            header = file[1]
            seq = file[2]

            item_dict = {}
            for item in site:
                sites = []
                for site_item in site[item]:
                    description = site_item["site_description"]
                    descriptions = description.split()
                    try:
                        sites.append(descriptions[1])
                    except IndexError :
                        #print(description)
                        print(pro_id)
                item_dict[item] = sites
            total_site.append(item_dict)
            #print(total_site)
    print("len(total_site)")
    print(len(total_site))
    with open("/home/zhaiyh884/20190614_new_data/total_site.pickle", "wb") as dbFile:
        pickle.dump(total_site, dbFile)
    pass

if __name__ == "__main__": 
    
    
    start = datetime.datetime.now()

    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0615_data"
        
    #get_site_info(rootdir,savefilepath)

    find_all_sites(savefilepath)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)     














































