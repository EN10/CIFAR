def write_cats():
    f = open('cats.txt','w')
    for i in range(0,10000):
        if (labelarray[i] == 2):
            f.write(str(i)+"\n")
    f.close()

def read_cats():
    f = open('cats.txt','r')
    text = f.read()
    print text
    f.close()