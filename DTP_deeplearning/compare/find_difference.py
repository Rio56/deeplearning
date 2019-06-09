import pickle
"""
data_old_pos_10290
data_old_neg_138822
data_new_pos_10291
data_new_neg_138808
"""
old_pos = open('data_old_pos_10290.pkl', 'rb')
old_neg = open('data_old_neg_138822.pkl', 'rb')

new_pos = open('data_new_pos_10291.pkl', 'rb')
new_neg = open('data_new_neg_138808.pkl', 'rb')

old_neg = pickle.load(old_neg)
print("old")
print(len(set(old_neg)))
old_neg = set(old_neg)

new_neg = pickle.load(new_neg)
print("new")
print(len(set(new_neg.keys())))
new_neg = set(new_neg.keys())
print(len(new_neg))

"""
print(len(old_neg & new_neg))
#print(old_neg & new_neg)
x = old_neg - new_neg
y = new_neg - old_neg
print(len(x))
print(len(y))
"""
count = 0

for item in old_neg:
    str = ""
    for aa in item:
        if aa == "_":
            aa = "X"
        str = str + aa


    if str not in new_neg:
        print("###")
        print(str)
        count = count + 1


for item in new_neg:
    str = ""
    for aa in item:
        if aa == "X":
            aa = "_"
        str = str + aa
    if str not in old_neg:
        print("$$$")
        print(str)

        count = count + 1

print(count)