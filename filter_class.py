from load_cifar import load

def write(labelarray):
    f = open('automobiles.txt','w')
    for i in range(0,10000):
        if (labelarray[i] == 1):
            f.write(str(i)+"\n")
    f.close()

def read():
    f = open('cats.txt','r')
    text = f.read()
    print text
    f.close()

la = load()
write(la)