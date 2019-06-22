import os
import sys
import numpy as np
import pickle
import math
from decimal import Decimal

class ParsePssm():
    
    def _loadFile(self,file): 
        ''' Open the file with the path
            @return: The lines of the file
            @param file: The full path of the file, str
        '''                 
        try: 
            with open(file) as fh:
                filelines = fh.readlines() #read file and get all lines
            return filelines 
        except IOError as err:
            print('File error: '+err)    
            
    def __parseonefilepssm(self,filelines):
        
        pssmvalue = []
        for line in filelines:    
            if len(line.split()) == 44:
                eachitem = line.split()[2:22]
                pssm = []
                for r in eachitem:
                    score = 1/(1+ math.exp(-float(r)))  
                    score = round(score,4)
                    pssm.append(score)
                pssmvalue.append(pssm)
        return pssmvalue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            
    def getpssmfeature(self,filepath):
    
        fileslist = os.listdir(filepath)
        allentrypssm = {}
        for onefile in fileslist:
            path = filepath + '/' + onefile
            filelines = self._loadFile(path)
            
            pssmvalue = self.__parseonefilepssm(filelines )
            
            uniprotID = onefile.split('/')[-1].split('.')[0]
            
            uniprotpssm = {uniprotID:pssmvalue}
            allentrypssm.update(uniprotpssm)
        return allentrypssm



if __name__ == '__main__':
    
    #filepath = "/home/gongjt057/drugsite/pssm"
    
    filepath = "/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/pssm3238/"
    pssm = ParsePssm()
    allentrypssm = pssm.getpssmfeature(filepath)
    #print(allentrypssm)
    print(len(allentrypssm))
    #for unp in allentrypssm:
        #print(allentrypssm[unp])
        #filepath = '/home/gongjt057/baoll/pssmpickle3/' + unp +'.pickle'
        #f = open(filepath,'wb')
        #pickle.dump(allentrypssm[unp], f, protocol=2)    
    pic = open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/PSSM_normalized_3234.pic','wb')
    pickle.dump(allentrypssm,pic,protocol=2)
