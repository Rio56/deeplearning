import os

class psiblast():
    def runblast(self,fastapath,outpath,pssmpath):
        names=[name for name in os.listdir(fastapath) if os.path.isfile(os.path.join(fastapath+'//', name))]
        for each_item in names:
            uniprotid=each_item.split('.')[0]
            cmd='/home/ThirdPartTools/blast/bin/psiblast -comp_based_stats 1 -evalue 0.001 -num_iterations 3 -db /home/ThirdPartTools/blast/db/swissprot -query '+fastapath+'/'+each_item+' -outfmt 0 -out '+outpath+'/'+uniprotid+'.fm0 -out_ascii_pssm '+pssmpath+'/'+uniprotid+'.pssm -num_threads 48'
            #print(cmd)
            os.system(cmd)
      
      
if __name__ == '__main__':
    fastapath = '/home/luchang/TMP/DATA/fasta'
    outpath = '/home/luchang/TMP/Medium_Data/Psiblast'
    pssmpath = '/home/luchang/TMP/Medium_Data/PSSM'
    psi = psiblast()
    psi.runblast(fastapath, outpath, pssmpath)