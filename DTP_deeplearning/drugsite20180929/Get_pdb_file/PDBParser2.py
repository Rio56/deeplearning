#coding:utf-8
'''
    Created on 2016-8-24

    @author: Gong Jianting

    Updated on 2016-9-21

    '''
import  logging,DBParser,datetime,ConfigDealer,pickle
import  logging,datetime,pickle
import json
import os
import unittest
#from Download import Download
import threading
from multiprocessing.dummy import Pool as ThreadPool 
import random
import time
from time import sleep 


########################################################################
#class PDBParser(DBParser.ParserBase):
class PDBParser():
    """

    ----------------------------------------------------------------------
    """
    def __init__(self,paraDic,targetDB):
        """Constructor"""
        #call the cpnstructor of its parent class
        pass
        #super().__init__(paraDic,targetDB)

        #overload the function of its parent class with a same name        
    def run(self):
        
        print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
        
    
        filehet = open('/home/RaidDisk/gongjt057/drugsite20180929/Get_Seq/Drug/hetlist.pickle','rb')
        hetlist =  pickle.load(filehet)
        
        #paraDic =  self.getConfigDict()
        
        rootdir = "/home/RaidDisk/gongjt057/drugsite20180929/Get_PDB_file/PDB_file/HET/"
        update_list = []
        for parent,dirnames,filenames in os.walk(rootdir):
            for filename in filenames:
                fileone = os.path.join(parent,filename)
                content = parse(fileone,hetlist)
                #update_list.append(os.path.join(parent,filename))     
        

                

        print(update_list)
        #updatelist_pik = self.parse_all_updateentry(update_list,paraDic)
        #print(len(updatelist_pik))
        
        print (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
        # do sth. new

    
    #------------------------------------------------------------------------------     
    #------------------------------------------------------------------------------ 
    
    def mkdir(self,path):
        #Created uncompress path folder
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path+"  is created")
            return True
        else:  
            #print(path )
            return False  
        

    #------------------------------------------------------------------------------ 
    def parse(self,PDBfile,hetlist):
        

        priStructure = self.getPriStructure(PDBfile)
        terStructure = self.getTerStructure(PDBfile)
        Sites = self.getSite(PDBfile)
        DBref = self.getDbref(PDBfile)

        content = self.final_format(terStructure,Sites,DBref,priStructure)        

        return content   
    #------------------------------------------------------------------------------
    
    def final_format(self,terStructure,Sites,DBref,priStructure):
        
        content = {}
        for key in terStructure:
            content[key] = {"tertiary_structure":{"PDB":{key:terStructure[key]}}}
            
            if key in priStructure:
                content[key].update({"Sequence":{"PDB":{key :priStructure[key]}}})
            else:
                pass
 
            if key in Sites:
                content[key].update({"Sites":{"PDB":{key:Sites[key]}}})
            else:
                pass 
            
            if DBref == {}:
                content[key].update({'IDs':{'PDB_chain_ID':key}})
            else:
                if key in DBref:
                    content[key].update(DBref[key])  
                else:
                    if "IDs" in content[key]:
                        content[key]["IDs"].update({'PDB_chain_ID': key})
                    else:
                        content[key].update({'IDs':{'PDB_chain_ID':key}})
                        
          
        return content
        
    #------------------------------------------------------------------------------
    def getDbref(self,PDBfile):
        
        lines = self.__loadPDBfile(PDBfile)
        DBref = {}
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':   
                pdb_id = self.__parsePDBIDLine(line)
            elif header == "DBREF" or header == "DBREF1":
                self.__parseDBrefline(line, DBref)
        
        for key in DBref.keys():
            try:
                unp = list(set(DBref[key]['IDs']['UNP']))
                DBref[key]['IDs']['UNP'] = unp
            except:
                DBref[key]['IDs']['UNP'] = []
        return DBref 

    #------------------------------------------------------------------------------
    
    def __parseDBrefline(self, line, DBref):
        
        chainid = line[7:11] + "_" + line[12]
        dbname = line[26:32].strip().upper()
        dbentryid = line[33:41].strip()
        dbentryname = line[42:54].strip()
        pdbchainbeg = line[14:18].strip()
        unichainbeg = line[55:60].strip()

        if chainid not in DBref :
            DBref.update({chainid:{"IDs":{'PDB_chain_ID':chainid,dbname: [ dbentryid ] }}})
        else:
            if dbname in DBref[chainid]["IDs"].keys():
                DBref[chainid]["IDs"][dbname].append(dbentryid)
            else:
                DBref[chainid]["IDs"].update({dbname:[dbentryid]})   
                
        return None
  
    #------------------------------------------------------------------------------
    
    def getSite(self,PDBfile):
        lines = self.__loadPDBfile(PDBfile)
        Sites = {}
        hetbindingsite = {}
        for g in range(0,len(lines)):
            line = lines[g]
            header = line.split()[0]
            if header == 'HEADER':   
                pdb_id = self.__parsePDBIDLine(line)
            elif header == "SITE":
                self.__parseSiteline(line , Sites, pdb_id)
            elif header == 'REMARK' and line.split()[1] == '800':
                if len(line.split()) >= 4:
                    if line.split()[2]  == 'SITE_IDENTIFIER:':
                        bindingsite = line.split()[3] 
                        description = ''
                        if lines[g+2].split()[2] == 'SITE_DESCRIPTION:':
                            if lines[g+2].split()[3].upper() == 'BINDING':
                                if len(lines[g+2].split()) >= 8:
                                    if lines[g+2].split()[5].upper() == 'FOR':
                                        description = lines[g+2].rstrip()[46:]
                                        if len(lines[g+3].split()) > 2:
                                            if lines[g+3].split()[0] == 'REMARK' and lines[g+3].split()[1] == '800' :
                                                if lines[g+3].split()[2] != "SITE" and lines[g+3].split()[2] != 'SITE_IDENTIFIER:':
                                                    description = description + lines[g+3].rstrip()
                                    hetbindingsite.update({bindingsite:description}) 
                                    #print(description)
                                
                            else:
                                description = line[29:].rstrip()                                   
                                if len(lines[g+3].split()) > 2:
                                    if lines[g+3].split()[0] == 'REMARK' and lines[g+3].split()[1] == '800' :
                                        if lines[g+3].split()[2] != "SITE" and lines[g+3].split()[2] != 'SITE_IDENTIFIER:' and lines[g+3].split()[2] != 'EVIDENCE_CODE:':
                                            description = description + lines[g+3].rstrip()  
                                hetbindingsite.update({bindingsite:description})
                                    #print(description)
    
                        else:
                            pass   
                    
        totalsite = self.formatSiteStructure(Sites,hetbindingsite)
        return  totalsite
    
    
    #------------------------------------------------------------------------------
    #def formatSiteStructure(self,Sites,hetbindingsite):
        #totalsite = {}
        #for chain in Sites.keys():   
            #chainsites = []
            #for sitename in Sites[chain].keys():
                #position = []
                #for residuesindex in  Sites[chain][sitename].keys():
                    #position.append(residuesindex)
                #if sitename in hetbindingsite.keys():
                    #chainsites.append({'position':position,"site_description":hetbindingsite[sitename]})
                #else:
                    #print(chain)
            #totalsite.update({chain:chainsites})  
        #return totalsite
    def formatSiteStructure(self,Sites,hetbindingsite):
        totalsite = {}
        for chain in Sites.keys():   
            chainsites = []
            for sitename in Sites[chain].keys():
                #position = []
                #for residuesindex in  Sites[chain][sitename].keys():
                    #position.append(residuesindex)
                if sitename in hetbindingsite.keys():
                    chainsites.append({'position':Sites[chain][sitename],"site_description":hetbindingsite[sitename]})
                else:
                    print(chain)
            totalsite.update({chain:chainsites})  
        return totalsite        
    

    
    #------------------------------------------------------------------------------

    def getPriStructure(self,PDBfile):
        lines = self.__loadPDBfile(PDBfile)
        primaryStr = {}
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':   
                pdb_id = self.__parsePDBIDLine(line)
            elif header == 'ATOM':
                self.__parsePriLine(line, primaryStr, pdb_id) 
        priStructure = self.formatPriStructure(primaryStr)
        return priStructure
        #return primaryStr
            
    #------------------------------------------------------------------------------

    def getTerStructure(self,PDBfile):
        lines = self.__loadPDBfile(PDBfile)
        tertiarystr = []
        for line in lines:
            header = line.split()[0]
            if header == 'HEADER':   
                pdb_id = self.__parsePDBIDLine(line)
            elif header == 'ATOM':
                self.__parseTerLine(line, tertiarystr , pdb_id)    
        terStructure = self.formatTerStructure(tertiarystr)
        return terStructure

    #------------------------------------------------------------------------------  
    
    def formatTerStructure(self, tertiarystr):
        terStructure = {}
        if tertiarystr != []:
            for i in range(0,len(tertiarystr)):
                terStructure[tertiarystr[i]["chain_name"]] = tertiarystr[i]         
        else:
            pass  
        return terStructure
    
    #------------------------------------------------------------------------------  
    
    def formatPriStructure(self, primaryStr):
        #print(primaryStr)
        priStructure = {}
        for pdbchain in primaryStr.keys():  
            priStructure.update(primaryStr[pdbchain])
        if "residue_index" in priStructure.keys():
            del priStructure["residue_index"]
        return priStructure   
    #------------------------------------------------------------------------------  


    def __parsePDBIDLine(self,line):
        """
        @return: The ChainID of PDBFile   
        @param 
        line: The line to be parsed
        chainid: The ChainID of PDBFile
        """  
        
        chainid = line[62:66].strip()
        return chainid 

    #------------------------------------------------------------------------------      

    def __loadPDBfile(self,fileName):
        '''
        @return: The lines of the PDB file
        @param pdbfile: The full path of the PDB file, str
        '''
        try: 
            with open(fileName) as fh:
                filelines = open(fileName).readlines()#read file and get all lines
                return filelines 
        except EnvironmentError as err:
            print(err)    
            
    #------------------------------------------------------------------------------

    def __parsePriLine(self, line, priStructure,pdb_id):
        """
        @return: None
        @param 
        line: The line to be parsed
        priStructure: The dict that record the primary structure(s) in the PDBFile
        """
        chain_name = pdb_id +"_" + line[21]
        atom_name = line[12:16].strip()
        residue_name = self.__transAA(line[17:20].strip())
        temp_factor = line[60:66].strip()
        resseq_position = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()


        if chain_name in priStructure.keys() :

            if int(resseq_position) -  int(priStructure[chain_name]['residue_index']) <= 0:
                pass
            else:      
                for j in range(0,(int(resseq_position) -  int(priStructure[chain_name]['residue_index'])) - 1  ):
                    priStructure[chain_name][chain_name] = priStructure[chain_name][chain_name] + "-"
                priStructure[chain_name][chain_name] = priStructure[chain_name][chain_name] + residue_name  
                priStructure[chain_name]['residue_index'] = resseq_position
        else: 
            priStructure.update({chain_name :{chain_name : residue_name , 'residue_index': resseq_position}})

        return None 
    
    #------------------------------------------------------------------------------

    def __parseTerLine(self, line,terStructure,pdb_id):
        """
        @return: None
        @param 
        line: The line to be parsed
        priStructure: The dict that record the tertiary structure(s) in the PDBFile

        """


        chain_name = pdb_id +"_" + line[21]
        atom_name = line[12:16].strip()
        residue_name = self.__transAA(line[17:20].strip())
        temp_factor = line[60:66].strip()
        resseq_position = line[22:26].strip()
        x = line[30:38].strip()
        y = line[38:46].strip()
        z = line[46:54].strip()



        label = True   

        for j in range(0,len(terStructure)):
            if terStructure[j]['chain_name'] == chain_name:
                elelen = len(terStructure[j]['residues'])
                label = False
                count = 0
                for i in range(0,elelen):
                    if terStructure[j]['residues'][i]['residue_name'] == residue_name and terStructure[j]['residues'][i]['residue_index'] == resseq_position :
                        terStructure[j]['residues'][i]['atoms'].append({'atom_name':atom_name,'atom_coordinates':[x,y,z],'atom_tempfactor':temp_factor})
                        count = 1
                        break                

                if count == 0:
                    terStructure[j]['residues'].append({'residue_name':residue_name,'residue_index':resseq_position,'atoms':[{'atom_name':atom_name,'atom_coordinates':[x,y,z],'atom_tempfactor':temp_factor}]}) 
                    sequence_end = terStructure[j]['chain_langth']['sequence_end']
                    if int(sequence_end) < int(resseq_position):
                        terStructure[j]['chain_langth']['sequence_end'] = resseq_position

        if label == True : 
            terStructure.append({'chain_name':chain_name,'chain_langth':{'sequence_begin':resseq_position,'sequence_end':resseq_position},'residues': [{'residue_name':residue_name,'residue_index':resseq_position,'atoms':[{'atom_name':atom_name,'atom_coordinates':[x,y,z],'atom_tempfactor':temp_factor}]}]})


        return None
    
    #------------------------------------------------------------------------------
    def __parseSiteline(self,line,Sites,pdb_id):

        chainid = pdb_id + "_"  
        site_name = line[11:14].strip()
        chain = {}
        if line[22] != ' ' or line[18:21].strip() != '' :
            if line[18:21].strip() == 'HOH' :
                pass
            else:
                chain = {chainid + line[22]: { line[23:27].strip(): self.__transAA(line[18:21].strip())} }
        if line[33] != ' ' or line[29:32].strip() != '' :
            if line[29:32].strip() == 'HOH':
                pass
            else:
                if (chainid + line[33]) not in chain:
                    chain.update({ chainid + line[33] :{line[34:38].strip() : self.__transAA(line[29:32].strip())} })
                else:
                    chain[chainid + line[33]].update({line[34:38].strip() : self.__transAA(line[29:32].strip())})            
        if line[44] != ' ' or line[40:43].strip() != '':
            if line[40:43].strip() == "HOH":
                pass
            else:
                if (chainid + line[44]) not in chain:
                    chain.update({chainid + line[44]:{line[45:49].strip():self.__transAA(line[40:43].strip() )} })
                else:
                    chain[chainid + line[44]].update({line[45:49].strip():self.__transAA(line[40:43].strip() )})                 
        if line[55] != ' ' or line[51:54].strip() != '' :
            if line[51:54].strip() == 'HOH':
                pass
            else:
                if (chainid +  line[55]) not in chain:
                    chain.update({chainid + line[55]:{line[56:60].strip():self.__transAA(line[51:54].strip())} })
                else:
                    chain[chainid + line[55]].update({line[56:60].strip():self.__transAA(line[51:54].strip())})                

        for key in chain:
            if key not in Sites:
                Sites.update({key:{site_name:chain[key]}})
            else:
                if site_name in Sites[key]:
                    Sites[key][site_name].update(chain[key])
                else:
                    Sites[key].update({site_name:chain[key]})     
        return None 
    
    #------------------------------------------------------------------------------

    def __transAA(self,x):
        """
        @return: The one-character amino acid name
        @param  x: The three-character amino acid name  
        """  
        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
             'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        if x in d.keys():
            return d[x]
        else:
            return 'X'


#if __name__ == "__main__": 
    #print('please input the path of the PDBfile:such as E:\\pdb\\pdb3rum.ent,E:\\pdb\\1a0s.pdb')
    #file=input()
    #pdbparser = PDBParser()
    ##content = pdbparser.parseTertiaryStructure(file)
    #content = pdbparser.getSite(file)  

    #formatinput = json.dumps(content, indent=1)
    #print(formatinput)
    #print("Done")


if __name__=="__main__":

    #logging.basicConfig(level=logging.INFO,                     
                        #format='%(thread)d %(threadName)s %(asctime)s %(levelname)s: %(message)s',
                        #datefmt='%a, %d %b %Y %H:%M:%S',
                        #filename='test.log',
                        #filemode='w')        
    #cfg = ConfigDealer.ConfigDealer()


    pdb=PDBParser()
    pdb.run()        