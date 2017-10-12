import cPickle
import numpy as np

path = 'cifar-10-batches-py/'
db1 = 'data_batch_1'
bm = 'batches.meta'

f = open(path+db1, 'rb')
dict = cPickle.load(f)
images = dict['data']
images = np.reshape(images, (10000, 3, 32, 32))
labels = dict['labels']
imagearray = np.array(images)
labelarray = np.array(labels)

f = open(path+bm, 'rb')
dict = cPickle.load(f)
categories= dict['label_names']

def output():
    print imagearray.shape
    print labelarray.shape
    print imagearray[0,0,0]
    print labelarray[0]
    
def classes():
    for i in range(0, 10):
        print str(i) + " : " + categories[i]

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
    
read_cats()