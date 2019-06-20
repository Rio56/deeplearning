# /home/RaidDisk/pdbfiles/updb
from PDBParseBase import PDBParserBase

import time, os, datetime, logging, gzip, pickle
import operator

def mkdir(path):
    # Created uncompress path folder
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + " Created folder sucessful!")
        return True
    else:
        # print ("this path is exist")
        return False


def get_site_info(rootdir, savefilepath):
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

    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            data_3_items = []
            # analyzedata

            pdb_site_info = pdbbase.get_site_info(os.path.join(parent, filename))
            data_3_items.append(pdb_site_info)
            if not pdb_site_info:
                print("do not have site" + filename)
                continue

            pdb_header_info = pdbbase.get_header_info(os.path.join(parent, filename))
            data_3_items.append(pdb_header_info)
            # print(pdb_header_info)

            pdb_seq_info = pdbbase.get_sequence_fromSEQ(os.path.join(parent, filename))
            data_3_items.append(pdb_seq_info)
            # print(pdb_seq_info)

            # save data
            if not pdb_site_info:
                pass

            else:
                dirname = filename[4:6]
                new_Filepath = savefilepath + "/" + str(dirname) + "/"
                mkdir(new_Filepath)
                new_filename = filename[3:7] + ".pickle"

                with open(new_Filepath + new_filename, "wb") as dbFile:
                    pickle.dump(data_3_items, dbFile)

                """with open(new_Filepath + new_filename,"rb") as dbFile:
                    file = pickle.load(dbFile)  """

                pass

    pass


def find_all_sites(rootdir):
    # use the pickles that contain  header\sequence\site\remark800 info
    
    total_site = []
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
                        sites.append(descriptions[1])
                    except IndexError:
                        print(pro_id)
                        print(description)
                        
                item_dict[item] = sites
            total_site.append(item_dict)
            # print(total_site)
    print("len(total_site)")
    print(len(total_site))
    with open("/home/zhaiyh884/20190614_new_data/total_site.pickle", "wb") as dbFile:
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

                total_site_num = total_site_num + 1
                if site_name in site_dicts.keys():
                    site_dicts[site_name] = site_dicts[site_name] + 1 if site_name in site_dicts else 1
                    print(site_name)
    with open("/home/zhaiyh884/20190614_new_data/site_numbers_sorted_2nd.pickle", "wb") as dbFile:
        pickle.dump(site_dicts, dbFile)


if __name__ == "__main__":
    start = datetime.datetime.now()

    rootdir = "/home/RaidDisk/pdbfiles/updb"
    savefilepath = "/home/zhaiyh884/20190614_new_data/0615_data"

    # get_site_info(rootdir,savefilepath)
    # find_all_sites(savefilepath)
    count_sites("/home/zhaiyh884/20190614_new_data/total_site.pickle")

    end = datetime.datetime.now()
    print("alltime = ")
    print (end - start)
