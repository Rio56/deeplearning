
import numpy as np
import pickle
import os


class AAindex(): 
    
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
            
    def getAAinexdict(self):
        pi = open('/home/RaidDisk/gongjt057/drugsite/AAindex/AAindexall.pic','rb')
        #pi = open('/home/RaidDisk/gongjt057/drugsite/AAindex/polar.pic','rb')
        aaindexdict = pickle.load(pi)
        return aaindexdict
        
    def GetAAindex(self,seq):
        
        aa = self.getAAinexdict()
        feature = []
        for each in seq:
            try:                
                feature.append(aa[each])
            except:
                if each == 'X':
                    #feature.append(['0','0','0','0','0']) #P35992 de 1631 
                    #print(uniprot)
                    pass       
        return feature


    def __parseonefile(self, filelines ):
        for i in range(0,len(filelines)):
            line = filelines[i]
            if line[0] == '>':
                uniprot = line[1:7]
                seq = filelines[i + 1].replace('\n','')
                #print(len(seq))
                feature = self.GetAAindex(seq)
            return feature


    def getaaindexfeature(self,filepath):
    
        fileslist = os.listdir(filepath)
        allentry = {}
        for onefile in fileslist:
            path = filepath + '/' + onefile
            filelines = self._loadFile(path)
            
            aaindexvalue = self.__parseonefile(filelines )
            
            uniprotID = onefile.split('/')[-1].split('.')[0]
            
            uniprotonehot = {uniprotID:aaindexvalue}
            allentry.update(uniprotonehot)
        return allentry  
    
    
    def getonehotfromtxt(self,filename):
        
        #aa = self.getAAinexdict()
        sequences = self._loadFile(filename)
        aaindexfeature = {}
        for i in range(0,len(sequences)):
            line = sequences[i]
            if line[0] == '>':
                uniprot = line[1:7]
                seq = sequences[i + 1].replace('\n','')
                #print(len(seq))
                feature = self.GetAAindex(seq)
                #print(feature)
                aaindexfeature.update({uniprot:feature})           
        return aaindexfeature
            
if __name__ == '__main__':
  
  
    
    filepath = "/home/luchang/TMP/DATA/fasta"
    aaindex = AAindex()
    aaindexfeature = aaindex.getaaindexfeature(filepath)
    print(len(aaindexfeature))
    pi = open('/home/luchang/TMP/Features/Aaindex_pickle.pic','wb')
    pickle.dump(aaindexfeature,pi,2) 

            
        