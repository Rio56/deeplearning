import os
import sys
import numpy as np
import pickle


class ParseTopology():
    
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
            
    def __parseonetopo(self,filelines):
        
        topovalue = []
        for line in filelines:    
            if line[0:10].strip() == 'pred' :
                print(line.replace('\n','')[10:])
                for each in line.replace('\n','')[10:]:
                    if each == ' ':
                        pass
                    else:
                        eachitem = self.__parseTopovalue(each)
                        topovalue.append(eachitem)
        return topovalue
    
    def __parseTopovalue(self,residuesvalue):
        
        if residuesvalue.upper() == 'O':
            return [0,0,1]
        elif residuesvalue.upper() == 'H':
            return [0,1,0]
        elif residuesvalue.upper() == 'I':
            return [1,0,0]
            
            
    def gettopofeature(self,filepath):
    
        fileslist = os.listdir(filepath)
        allentrytopo = {}
        for onefile in fileslist:
            path = filepath + '/' + onefile
            filelines = self._loadFile(path)
            topovalue = self.__parseonetopo(filelines )
            uniprotID = onefile.split('/')[-1].split('.')[0]
            uniprotentry = {uniprotID:topovalue}
            allentrytopo.update(uniprotentry)
        return allentrytopo



if __name__ == '__main__':
    
    filepath = "//home/luchang/TMP/Medium_Data/Topology"
    
    #filepath = "/home/gongjt057/drugsite/hmmtoptest"
    topo = ParseTopology()
    allentrytopo = topo.gettopofeature(filepath)
    
    print(allentrytopo)
    pic = open('/home/luchang/TMP/Features/Topology_pickle.pic','wb')
    pickle.dump(allentrytopo,pic,2)    