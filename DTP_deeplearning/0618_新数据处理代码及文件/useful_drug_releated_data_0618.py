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


def get_drug_releated_info(rootdir,savefilepath):
    """extract header\sequence\site\remark800 info in rootdir.
    and then, save them as a pickle content with list[1site,header,seq_from_SEQ,seq_from_ATOM,atom_info]
    
    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/????"
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
            data_5_items = []
            #data_5_items[1site,header,seq_from_SEQ,seq_from_ATOM,atom_info]
            #analyzedata
            pdb_site_info = pdbbase.get_site_info(os.path.join(parent,filename))
            #print(pdb_site_info)
            data_5_items.append(pdb_site_info)          
            if not pdb_site_info :
                print("no site:" + filename)
                "in order to save some time"
                continue
            
            pdb_header_info = pdbbase.get_header_info(os.path.join(parent,filename))
            data_5_items.append(pdb_header_info)
            #print(pdb_header_info)      
            
            pdb_seq_from_SEQ__info = pdbbase.get_sequence_fromSEQ(os.path.join(parent,filename))
            data_5_items.append(pdb_seq_from_SEQ__info)
            #print("pdb_seq_from_SEQ__info")
            #print(pdb_seq_from_SEQ__info)        
            
            pdb_seq_from_ATOM_info = pdbbase.get_sequence_fromATOM(os.path.join(parent,filename))
            data_5_items.append(pdb_seq_from_ATOM_info)
            #print("pdb_seq_from_ATOM_info")
            #print(pdb_seq_from_ATOM_info) 

        
            pdb_atom_info_info = pdbbase.get_atom_info(os.path.join(parent,filename))
            for items in pdb_atom_info_info:   
                pdb_atom_info_info[items].pop('residues')
            data_5_items.append(pdb_atom_info_info)

                
            #save data
            if not pdb_site_info :
                pass                                         
            else:
                dirname = filename[4:6]       
                new_Filepath = savefilepath +"/" +  str(dirname)+"/"
                mkdir(new_Filepath)  
                new_filename =  filename[3:7] + ".pickle"      
                
                with open(new_Filepath + new_filename,"wb") as dbFile:
                    pickle.dump(data_5_items,dbFile)         
                """with open(new_Filepath + new_filename,"rb") as dbFile:
                    file = pickle.load(dbFile)  """
                pass
    pass

def get_lable_and_seqs(rootdir):
    # use the pickles that contain t[1site,header,seq_from_SEQ,seq_from_ATOM,atom_info]
    #rootdir =  "/home/zhaiyh884/20190614_new_data/0618_drug_related_protein_data"
    #
    
    pic = open('/home/zhaiyh884/20190614_new_data/hetlist.pickle','rb')
    hetlist = pickle.load(pic)    
    #print(len(hetlist))
    #print(hetlist)
    
    
    total_site = 0
    description_null = 0
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # walk into one file

            pro_id = filename[0:4]
            file_path = os.path.join(parent, filename)
            with open(file_path, "rb") as dbFile:
                file = pickle.load(dbFile)
            # print(file)
            site_info = file[0]
            header_info = file[1]
            seq_from_SEQ_info = file[2]
            seq_from_ATOM_info = file[3]
            atom_info = file[4]
            #print("###############################")
            #print(site_info)
            #print(site_info)
            #print(header_info)
            #print(seq_from_SEQ_info)
            #print(seq_from_ATOM_info)
            #print(atom_info)

            #>ABCD_A
            #seq_from_SEQ_info
            #seq_from_ATOM_info
            #lable
            
            for chain_id in site_info:
                # in a cetern chain
                for dif_bindings in site_info[chain_id]:
                    #one binding in many different bindings
                    binding_position = {}
                    whole_description = dif_bindings["site_description"]
                    site_description = whole_description.split()
                    try:
                        #find the binding_object name
                        if "res" in site_description[1]:
                            binding_object = site_description[2]
                        elif "RESIDUES" in whole_description and "THROUGH" in whole_description:
                            binding_object = site_description[2]
                        else:
                            binding_object = site_description[1]    
                    except IndexError:
                        #print(header_info)
                        #print(whole_description)
                        description_null = description_null+1     
                        
                    if binding_object in hetlist and binding_object !="GOL":
                        #if it is drug releated binding.
                        binding_position.update(dif_bindings["position"])
                        pass
                num = len(binding_position)
                
                samelist = []
                try:
                    pdbbegin = atom_info[chain_id]["chain_langth"]["sequence_begin"]
                    pdbsequence = seq_from_ATOM_info[chain_id]
                    pdb_seq_from_SEQ = seq_from_SEQ_info[chain_id]
                    pass
                except KeyError:
                    if num> 0:
                        print(header_info)
                        print(atom_info)
                        print(site_info)
                        print(seq_from_ATOM_info)

                for indexkey in binding_position.keys():
                    seqindex = int(indexkey) - int(pdbbegin)
                    if seqindex < len(pdbsequence)  and seqindex  > 0:
                        if binding_position[indexkey] == pdbsequence[seqindex]:
                            samelist.append(seqindex)
                            pass
                        pass
                    pass
                
                if num == 0:
                    pass
                else:
                    #if len(samelist)/num > 0.25:
                    if len(pdbsequence)>50:
                        print('>' + chain_id)
                        print(pdb_seq_from_SEQ)
                        print(pdbsequence)
                        site = ''
                        for i in range(0, len(pdbsequence)):
                            if i in samelist:
                                site = site + '1'
                                total_site = total_site+1
                            else:
                                site = site + '0'
                        print(site)
                        pass



            """

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
    pass"""

    print("total_site:")
    print(total_site)
    print("done")















if __name__ == "__main__": 

    start = datetime.datetime.now()
    
    #1 extract all needed infomation from pdb
    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0618_drug_related_protein_data"
    #get_drug_releated_info(rootdir,savefilepath)
    
    #2 get lables
    get_lable_and_seqs(savefilepath)

    end = datetime.datetime.now()
    print("alltime = ")
    print (end-start)         
