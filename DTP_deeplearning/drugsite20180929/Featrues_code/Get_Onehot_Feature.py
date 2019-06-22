
import numpy as np
import pickle
import os


class Profile(): 
    
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

    def __tranAindex(self,res):
        profiledict = {'A':1,'R':2,'D':3,'C':4,'Q':5,
                       'E':6,'H':7,'I':8,'G':9,'N':10,
                       'L':11,'K':12,'M':13,'F':14,'P':15,
                       'S':16,'T':17,'W':18,'Y':19,'V':0} 
        if res in profiledict:
            return profiledict[res]
        else:
            return ''
        
    def GetProfile(self,seq,array = True):

        rownumber = 20
        proMatr = np.zeros((len(seq),20))
        
        for i in range(0,len(seq)):
            each = seq[i]
            row = i
            col = self.__tranAindex(each)
            
            if col != '':
                proMatr[row][col] = 1
            else:
                pass
                #proMatr[row][20] = 1
        if array == True:
            return proMatr
        else:
            return np.ndarray.tolist(proMatr )

    def __parseonefile(self, filelines ):
        for i in range(0,len(filelines)):
            line = filelines[i]
            if line[0] == '>':
                uniprot = line[1:7]
                seq = filelines[i + 1].replace('\n','')
                #print(len(seq))
                feature = self.GetProfile(seq,False)
            return feature


    def getonehotfeature(self,filepath):
    
        fileslist = os.listdir(filepath)
        allentryonehot = {}
        for onefile in fileslist:
            path = filepath + '/' + onefile
            filelines = self._loadFile(path)
            
            onehotvalue = self.__parseonefile(filelines )
            
            uniprotID = onefile.split('/')[-1].split('.')[0]
            
            uniprotonehot = {uniprotID:onehotvalue}
            allentryonehot.update(uniprotonehot)
        return allentryonehot  
    
    
    def getonehotfromtxt(self,filename):
        
        sequences = self._loadFile(filename)
        One_Hotfeature = {}
        for i in range(0,len(sequences)):
            line = sequences[i]
            if line[0] == '>':
                uniprot = line[1:7]
                seq = sequences[i + 1].replace('\n','')
                #print(len(seq))
                feature = self.GetProfile(seq,False)
                #print(feature)
                One_Hotfeature.update({uniprot:feature})           
        return allonehotfeature
            
if __name__ == '__main__':
  
  
    
    filepath = "/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/fasta3238"
    onehot = Profile()
    One_Hotfeature = onehot.getonehotfeature(filepath)
    print(One_Hotfeature)
    print(len(One_Hotfeature))
    pi = open('/home/RaidDisk/gongjt057/drugsite20180929/Mi_Features/Onehot3238.pic','wb')
    pickle.dump(One_Hotfeature,pi,2) 
