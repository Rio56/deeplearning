with open('test_5_gpu7_011721.out', 'r') as f:
    #print(f.readlines())
    lines  = f.readlines()
    line_number = 0
    for line in lines:
        #print(line[0:2] )
        #if line[0:2] == '[0'
        if 'val_loss'in line:
            line_number = line_number + 1
            #print(line)
            loss = line[68:75]
            acc = line[82:89]
            val_loss = line[101:108]
            val_acc = line[119:127]
            str = loss + "#" + acc + "#" + val_loss + "#" + val_acc

            if line_number % 10 == 0:
                pass
                with open('test_5_gpu7_011721_acc&loss.txt', 'a') as f:
                    f.write(str)
                    #f.write('\n')
                f.close

        if line[0:2] == '[0' and len(line) >50:

            with open('test_5_gpu7_011721_others.txt', 'a') as f:
                f.write(line)
        
    

