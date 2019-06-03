
import pickle
file = open("0603_train.fasta")

seqid = []
for line in file:

    if ">" in line:
        print(line)
        print(line[1:7])
        seqid.append(line[1:7])
    pass # do something
file.close()
print(seqid)

f = open("0603_train.pickle","wb")
pickle.dump(seqid, f)

training_data_id = open("seq_cut_0.9.pickle", 'rb')
training_data_id = pickle.load(training_data_id)
print(training_data_id)
print(len(training_data_id))

training_data_id = open("0603_train.pickle", 'rb')
training_data_id = pickle.load(training_data_id)
print(training_data_id)
print(len(training_data_id))

