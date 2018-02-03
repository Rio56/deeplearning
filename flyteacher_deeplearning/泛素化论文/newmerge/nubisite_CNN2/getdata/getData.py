import pandas as pd
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
#import keras.models as models
#from keras.models import Model
#from methods.DProcess import convertRawToXY

 
def getsitepssm(seqpssm,i,win):
    if win <= i  and  len(seqpssm) - win -1 >= i :
        return seqpssm[i-win :i+win+1]  
    elif win > i and len(seqpssm) - win - 1 >= i :
        #left short
        pssm = []
        for j in range(win-i):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        pssm.extend(seqpssm[:i+win+1])
        return pssm
    elif win <= i and len(seqpssm) - 1 - win < i :
        #right short
        pssm = []
        pssm = seqpssm[i-win :len(seqpssm)]
        for j in range(win+i - len(seqpssm) + 1):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        return pssm
    else: 
        #left and right short
        pssm = []
        for j in range(win-i):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])        
        pssm.extend(seqpssm)
        for j in range(win+i - len(seqpssm) + 1):
            pssm.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])   
        return pssm 
    
def find(i,id_num):
    for j in range(len(id_num)):
        if (id_num[j]==i):
            return True
    return False

def subSeqq(seq,id,num):
    win = num
    subSeq = ''
    if (id-win-1)<0 and (id + win+1)>len(seq):
        for i in range(win-id+1):
            subSeq+='B'
        for i in range(0,len(seq)-id-1):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='B'
    elif (id-win-1)<0 and (id+win+1)<=len(seq):
        for i in range(win-id+1):
            subSeq+='B'
        for i in range(0,id+win+1-1):
            subSeq+=seq[i]
    elif (id-win-1)>=0 and (id+win+1) > len(seq):
        for i in range(id-win-1,len(seq)-1):
            subSeq+=seq[i]
        for i in range(len(seq),id+win+1):
            subSeq+='B'
    elif (id-win-1)>=0 and (id+win+1) <= len(seq):
        for i in range(id-win-1,id+win+1-1):
            subSeq+=seq[i]
    return subSeq    

def get_global_pssm(sequence_fragment,pssm_frgmant):
    
    AAcid_index = {}  
    AAcid_index["A"] = []
    AAcid_index["C"] = []
    AAcid_index["D"] = []
    AAcid_index["E"] = []
    AAcid_index["F"] = []
    AAcid_index["G"] = []
    AAcid_index["H"] = []
    AAcid_index["I"] = []
    AAcid_index["K"] = []
    AAcid_index["L"] = []
    AAcid_index["M"] = []
    AAcid_index["N"] = []
    AAcid_index["P"] = []
    AAcid_index["Q"] = []
    AAcid_index["R"] = []
    AAcid_index["S"] = []
    AAcid_index["T"] = []
    AAcid_index["V"] = []
    AAcid_index["W"] = []
    AAcid_index["Y"] = []
    #AAcid_index["U"] = []
    #AAcid_index["Z"] = []
    #AAcid_index["B"] = []
    
    results = []
    
    for i in range(0,len(sequence_fragment)):
        AA = sequence_fragment[i]
        if not AA in AAcid_index:
            continue
        else:
            AAcid_index[sequence_fragment[i]].append(i)
    
    for key,value in AAcid_index.items():
        temp_vector = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        if len(value) != 0:
            for i in range(0,len(value)):
                temp_vector = [temp_vector[j] + pssm_frgmant[value[i]][j] for j in range(0, len(temp_vector))]
            temp_vector = [float(temp_vector[j]) /  len(value) for j in range(0, len(temp_vector))]
        results.append(temp_vector)
    return results


def read_fasta(fasta_file, windown_length,pssmfilepath):
    oneofkey_pos = []
    oneofkey_neg = []
    pssm_pos = []
    pssm_neg = []
    global_pssm_pos = []
    global_pssm_neg = []
    num = 0
    id_num=[]
    #pic = open('/home/gongjt057/baoll/pssmfeat.pickle','rb')
    #pssmfea = pickle.load(pic)  
    
    with open(fasta_file,'r') as fp:

        win = windown_length
        neg = 0
        pos = 0
        aseqpssm = ''
        wrr = 0
        for line in fp:
            num += 1
            if (num == 2):
                line1 = line.split('\t')
                name1 = line1[1]
                id = int(line1[2])
                seq1 = line1[4]
                id_num.append(id-1)
                del line1
            elif (num > 2):
                line1 = line.split('\t')
                name = line1[1]
                seq = line1[4]
                
                if (name == name1):
                    id_num.append(int(line1[2])-1)
                    del line1
                else:
                    del aseqpssm
                    aseqpssm = getpssmfeature(name1,pssmfilepath)
                    for i in range(len(seq1)):
                        if (seq1[i]=='K' and find(i,id_num)):
                            #print(i)
                            wrr+=1
                            pos += 1
                            subSeq = subSeqq(seq1,i,win)
                            final_seq = [1] + [AA for AA in subSeq]
                            oneofkey_pos.append(final_seq)
                            #add
                            onesitepssm = getsitepssm(aseqpssm,i,win)
                            pssm_pos.append(onesitepssm)
                            global_pssm_pos.append(get_global_pssm(
                                subSeq,onesitepssm))
                            del onesitepssm,subSeq,final_seq
                        elif (seq1[i]=='K' and not find(i,id_num)):
                            neg += 1
                            subSeq = subSeqq(seq1,i,win)
                            final_seq = [0] + [AA for AA in subSeq]  
                            oneofkey_neg.append(final_seq)
                            #add
                            onesitepssm = getsitepssm(aseqpssm,i,win)
                            pssm_neg.append(onesitepssm)
                            global_pssm_neg.append(get_global_pssm(subSeq,onesitepssm))                            
                            del onesitepssm,subSeq,final_seq
                    #print(wrr)
                    wrr = 0
                    id_num = []
                    name1 =  name
                    seq1 = seq
                    id_num.append(int(line1[2])-1)
                    del line1
        for i in range(len(seq1)):
            
            if (seq1[i]=='K' and find(i,id_num)):
                #add
                onesitepssm = getsitepssm(aseqpssm,i,win)
                pssm_pos.append(onesitepssm)               
                pos += 1
                subSeq = subSeqq(seq1,i,win)
                final_seq = [1] + [AA for AA in subSeq]
                oneofkey_pos.append(final_seq)
                global_pssm_pos.append(get_global_pssm(
                    subSeq,onesitepssm))                
                del onesitepssm,subSeq,final_seq
            elif (seq1[i]=='K' and not find(i,id_num)):
                #add
                onesitepssm = getsitepssm(aseqpssm,i,win)
                pssm_neg.append(onesitepssm)                
                neg += 1
                subSeq = subSeqq(seq1,i,win)
                final_seq = [0] + [AA for AA in subSeq]  
                oneofkey_neg.append(final_seq) 
                global_pssm_neg.append(get_global_pssm(subSeq,onesitepssm))   
                del onesitepssm,subSeq,final_seq
        del aseqpssm
        #print(wrr)
        #print(num)
        print(pos,' ',neg)
        #print(pssm_feature)
        return oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,global_pssm_pos,global_pssm_neg
                            
def getpssmfeature(unpac,pssmfilepath):
    pssmpicklepath = pssmfilepath + unpac + '.pickle'
    picklefile = open(pssmpicklepath,'rb')
    oneseqpssm = pickle.load(picklefile)
    picklefile.close()
    return oneseqpssm
    
def get_data(string,pssmfilepath):
    
    #string = '/home/gongjt057/baoll/Ubisite_test.txt'
    
    #pssmfilepath = '/home/gongjt057/baoll/pssmpickle/'
    winnum = 24
    # train dataset
    oneofkey_pos,oneofkey_neg,pssm_pos,pssm_neg,global_pssm_pos,global_pssm_neg = read_fasta(string,winnum,pssmfilepath)
    #print(len(pssm_feature) )

    oneofkey_pos = pd.DataFrame(oneofkey_pos)
    oneofkey_pos = oneofkey_pos.as_matrix()
    
    oneofkey_neg = pd.DataFrame(oneofkey_neg)
    oneofkey_neg = oneofkey_neg.as_matrix()
    
    
    
    return pssm_pos,pssm_neg,oneofkey_pos,oneofkey_neg,global_pssm_pos,global_pssm_neg

    

if __name__ == '__main__':
    g = get_data()
    
