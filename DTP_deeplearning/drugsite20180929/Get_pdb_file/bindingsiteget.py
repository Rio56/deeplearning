import pickle
import os


pic = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_Seq/Drug/hetlist.pickle','rb')
hetlist = pickle.load(pic)
#hetlist = ['K']

rootdir = '/home/RaidDisk/gongjt057/drugsite20180929/Get_PDB_file/PDB_parse_file/'
picklefiles = []
for parent,dirnames,filenames in os.walk(rootdir): 
    for filename in filenames:
        picklefiles.append(os.path.join(parent,filename))
        
count = 0
siteDic = {}
for item in picklefiles:
    it = open(item,'rb')
    dic = pickle.load(it)
    chainid = item.split('/')[-1].split('.')[0]
    pdbbegin = dic["tertiary_structure"]['PDB'][chainid]["chain_langth"]["sequence_begin"]
    pdbsequence = dic['Sequence']['PDB'][chainid]
    if 'Sites' in dic:
        sitename = dic["Sites"]['PDB'][chainid]
        print(sitename)
        position = {}
        for i in range(0,len(sitename)):
            sitesdes = sitename[i]["site_description"].split()
            for each in sitesdes:
                if each in hetlist:
                    position.update(sitename[i]["position"])
                    break
        
        num = len(position) 
        samelist = []
        for indexkey in position.keys():
            seqindex = int(indexkey) - int(pdbbegin)
            if seqindex < len(pdbsequence)  and seqindex  > 0:
                if position[indexkey] == pdbsequence[seqindex]:
                    samelist.append(seqindex)
            #else:
                #print(position)
        if num == 0:
            pass
        else:
            #if len(samelist)/num > 0.25:
            print('>'+chainid)
            print(pdbsequence)
            siteDic[chainid] = samelist
            site = ''
            for i in range(0,len(pdbsequence)):
                if i in samelist:
                    site = site + '1'
                else:
                    site = site + '0'
                print(site)
sitePickle = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_Site_pickle/drugsite.pickle','wb')
pickle.dump(siteDic,sitePickle)
#print(siteDic)
                        
                
            
            
            
            
        