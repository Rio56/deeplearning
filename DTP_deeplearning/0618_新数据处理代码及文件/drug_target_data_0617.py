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
    pdb_seq_info = pdbbase.get_sequence_fromATOM('/home/RaidDisk/pdbfiles/updb/pdb/a2/pdb2a2q.ent')
    print(pdb_seq_info)
    print(pdb_eq_info)
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
            
            pdb_seq_from_SEQ__info = pdbbase.get_sequence_fromSEQ(os.path.join(parent,filename))
            data_3_items.append(pdb_seq_from_SEQ__info)
            print("pdb_seq_from_SEQ__info")
            print(pdb_seq_from_SEQ__info)        
            
            pdb_seq_from_ATOM_info = pdbbase.get_sequence_fromATOM(os.path.join(parent,filename))
            data_3_items.append(pdb_seq_from_ATOM_info)
            print("pdb_seq_from_ATOM_info")
            print(pdb_seq_from_ATOM_info) 

            
            
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

def find_all_sites(rootdir):
    # use the pickles that contain  header\sequence\site\remark800 info
    #rootdir =  "/home/zhaiyh884/20190614_new_data/0615_data"
    #
    
    total_site = []
    description_null = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # walk into one file

            pro_id = filename[0:4]
            file_path = os.path.join(parent, filename)
            with open(file_path, "rb") as dbFile:
                file = pickle.load(dbFile)
            # print(file)
            site = file[0]
            header = file[1]
            seq = file[2]
            
            item_dict = {}
            for item in site:
                #{'1KMH_A': [{'position': {'51': 'G', '65': 'L', '131': 'E', '274': 'M', '297': 'R'}, 'site_description': 'RESIDUE TTX B 499'}],
                #        '1KMH_B': [{'position': {'81': 'A', '82': 'T', '83': 'D'}, 'site_description': 'RESIDUE TTX B 499'}]}                
                #in this loop item means protein_sequence name: 1KMH_A
                sites = []
                for site_item in site[item]:                 
                    #site_item means every record in one sequence
                    description = site_item["site_description"]
                    descriptions = description.split()
                    try:
                        #find the binding name
                        if "res" in descriptions[1]:
                            sites.append(descriptions[2])
                        else:
                            sites.append(descriptions[1])
                    except IndexError:
                        description_null = description_null+1
                item_dict[item] = sites
            total_site.append(item_dict)
            # print(total_site)
    print("len(membrane_total_site):")
    print(len(total_site))
    print("null_discription:")
    print(description_null)
    
    with open("/home/zhaiyh884/20190614_new_data/total_site.pickle", "wb") as dbFile:
        pass
        pickle.dump(total_site, dbFile)
    pass

def count_sites(file):
    #used to count sites and analyzedata.
    #file = "/home/zhaiyh884/20190614_new_data/total_site.pickle"
    #scan all the data and count every site's apperance
    
    with open(file, "rb") as dbFile:
        file = pickle.load(dbFile)
    site_dicts = {}
    total_site_num = 0
    for item in file:
        for seq_id in item:
            #print(seq_id)
            for site_name in item[seq_id]:
                print(site_name)
                
                site_dicts[site_name] = site_dicts[site_name] + 1 if site_name in site_dicts else 1

                total_site_num = total_site_num + 1
                #if site_name in site_dicts.keys():
                 #   site_dicts[site_name] = site_dicts[site_name] + 1 if site_name in site_dicts else 1
                    #print(site_name)

    with open("/home/zhaiyh884/20190614_new_data/site_numbers.pickle", "wb") as dbFile:
        pickle.dump(site_dicts, dbFile)
    print(site_dicts)
    
    print("total_site_num:")       
    print(total_site_num)
    print("site_dicts items num:")       
    print(len(site_dicts))    
    


        
def sites_anylize():
    with open("site_numbers.pickle","rb") as dbFile:
        file = pickle.load(dbFile) 
        
    with open("hetlist.pickle","rb") as dbFile_drug:
        file_drug = pickle.load(dbFile_drug)    
        
    sites_number = 0
    number_counter = {}
    
    drug_site = {}
    for site_name in file:

        if site_name in file_drug:
            sites_number = sites_number + file[site_name]
            # used to count all numbers of drugs_binding object
            number_of_site = file[site_name]
            # number_of_site used to sign the numbers which apperence
            number_counter[number_of_site] = number_counter[number_of_site] + 1 if number_of_site in  number_counter else 1     
            # the dict to store the number of times 
            drug_site[site_name] = file[site_name]
            #form a new site of drug sites
            
    print(sites_number) 
    print(number_counter)
    print(sorted(file.items(),key=lambda x:x[1]))    
    print("#@!#!$!@#%!#%")
    print(sorted(drug_site.items(),key=lambda x:x[1]))    
    pass


def find_memberain_sites(rootdir):
    # use the pickles that contain  header\sequence\site\remark800 info
    #rootdir =  "/home/zhaiyh884/20190614_new_data/0615_data"
    #
    
    total_site = []
    description_null = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # walk into one file

            pro_id = filename[0:4]
            file_path = os.path.join(parent, filename)
            with open(file_path, "rb") as dbFile:
                file = pickle.load(dbFile)
            # print(file)
            site = file[0]
            header = file[1]
            seq = file[2]

            #select membrane protein
            if "MEM" not in header["HEADER_classification"]:
                continue

            # use site info only
            item_dict = {}
            for item in site:
                #{'1KMH_A': [{'position': {'51': 'G', '65': 'L', '131': 'E', '274': 'M', '297': 'R'}, 'site_description': 'RESIDUE TTX B 499'}],
                #        '1KMH_B': [{'position': {'81': 'A', '82': 'T', '83': 'D'}, 'site_description': 'RESIDUE TTX B 499'}]}                
                #in this loop item means protein_sequence name: 1KMH_A
                sites = []
                for site_item in site[item]:                 
                    #site_item means every record in one sequence
                    description = site_item["site_description"]
                    descriptions = description.split()
                    try:
                        #find the binding name
                        
                        if "res" in descriptions[1]:
                            sites.append(descriptions[2])
                        else:
                            sites.append(descriptions[1])
                            
                    except IndexError:
                        description_null = description_null+1

                item_dict[item] = sites
            total_site.append(item_dict)
            # print(total_site)
    print("len(membrane_total_site):")
    print(len(total_site))
    print("null_discription:")
    print(description_null)
    
    with open("/home/zhaiyh884/20190614_new_data/membrane_total_site.pickle", "wb") as dbFile:
        pass
        pickle.dump(total_site, dbFile)
    pass


def find_drug_releated_protein(rootdir):
    # use the pickles that contain  header\sequence\site\remark800 info
    #rootdir =  "/home/zhaiyh884/20190614_new_data/0615_data"
    #find the difference between proteins class and drug-releated-proteins class
    
    with open("hetlist.pickle","rb") as dbFile_drug:
        file_drug = pickle.load(dbFile_drug)    
        
    protein_classfication = []
    drug_protein_classfication = []
    
    protein_dict = {}
    drug_releated_protein_dict = {}    
    
    description_null = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # walk into one file

            pro_id = filename[0:4]
            file_path = os.path.join(parent, filename)
            with open(file_path, "rb") as dbFile:
                file = pickle.load(dbFile)
            # print(file)
            site = file[0]
            header = file[1]
            seq = file[2]

            classification = header["HEADER_classification"]
            protein_dict[classification] = protein_dict[classification] + 1 if classification in  protein_dict else 1     
            #print(protein_dict)
            drug_releated_protein_flag = 0

            for item in site:
                #{'1KMH_A': [{'position': {'51': 'G', '65': 'L', '131': 'E', '274': 'M', '297': 'R'}, 'site_description': 'RESIDUE TTX B 499'}],
                #        '1KMH_B': [{'position': {'81': 'A', '82': 'T', '83': 'D'}, 'site_description': 'RESIDUE TTX B 499'}]}                
                #in this loop item means protein_sequence name: 1KMH_A
                sites = []
                for site_item in site[item]:                 
                    #site_item means every record in one sequence
                    description = site_item["site_description"]
                    descriptions = description.split()
                    try:
                        #find the binding_object name
                        if "res" in descriptions[1]:
                            binding_object = descriptions[2]
                        elif "RESIDUES" in description and "THROUGH" in description:
                            binding_object = descriptions[2]
                        else:
                            binding_object = descriptions[1]
                    except IndexError:
                        description_null = description_null+1
                    #sites.append(binding_object)
                    
                    if binding_object in file_drug:
                        drug_releated_protein_flag = 1
                        
            #print(drug_releated_protein_flag)
            if drug_releated_protein_flag ==1:
                drug_releated_protein_dict[classification] = drug_releated_protein_dict[classification] + 1 if classification in  drug_releated_protein_dict else 1

        """item_dict[item] = sites
            total_site.append(item_dict)
            # print(total_site)
    print("len(membrane_total_site):")
    print(len(total_site))
    print("null_discription:")
    print(description_null)"""
    print(protein_dict)
    print("!@#$!@#################$!@%#!$^%$#@^$%&^#$%&")
    print(drug_releated_protein_dict)
    with open("/home/zhaiyh884/20190614_new_data/drug_and_nondrug_protein_classfication.pickle", "wb") as dbFile:
        pass
        pickle.dump(protein_dict, dbFile)
        pickle.dump(drug_releated_protein_dict, dbFile)
    pass



if __name__ == "__main__": 

    start = datetime.datetime.now()
    
    #1 extract all needed infomation from pdb
    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0615_data"
    get_site_header_seq_info(rootdir,savefilepath)
    
    
    #rootdir = "/home/zhaiyh884/20190614_new_data/0615_data"
    #2 find_all_site
    #find_all_sites(rootdir)    
    #2 or find_memberain_sites
    #find_memberain_sites(rootdir)
    
    
    #3 count site numbers  
    #file = "/home/zhaiyh884/20190614_new_data/membrane_total_site.pickle"
    #file = "/home/zhaiyh884/20190614_new_data/total_site.pickle"
    #count_sites(file)
    
    
    #4
    #sites_anylize()
    
    
    #5
    #rootdir = "/home/zhaiyh884/20190614_new_data/0615_data"
    #find_drug_releated_protein(rootdir)




    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)     