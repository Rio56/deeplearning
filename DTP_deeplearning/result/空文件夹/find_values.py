with open('curve5_0114.out', 'r') as f:
    #print(f.readlines())
    lines  = f.readlines()
    for line in lines:
        #print(line[0:2] )
        #if line[0:2] == '[0'
        if 'val_loss'in line:
            #print(line)
            
            loss = line[68:75]
            acc = line[82:89]
            val_loss = line[101:108]
            val_acc = line[119:126]
            str = loss + " " + acc + " " + val_loss + " " + val_acc
            #print(str)
            
            
            with open('5_loss_acc.txt', 'a') as f:
                pass
                f.write(str)            
        
    

