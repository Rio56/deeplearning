#20181112 gong


f = open ('/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/seq_from_pdb_all_no_.fasta','r').readlines()
print(len(f))
for i in range(len(f)):
    if not len(f[i]):
        continue
    line = f[i]
    if line[0] == '>':
        
        uni = line.strip('\n')[1:]
        seq = f[i+1].strip()
        f1 = open ('/home/RaidDisk/gongjt057/drugsite20180929/DATA_sets_1/fasta/'+uni+'.fasta','w')
        f1.write(line)
        f1.write(seq)      
