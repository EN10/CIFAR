def load():
    import cPickle
    import numpy as np

    path = 'cifar-10-batches-py/'
    db1 = 'data_batch_1'

    f = open(path+db1, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    #images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = np.array(images)
    labelarray = np.array(labels)

def output():
    print imagearray.shape
    #print labelarray.shape

load()
output()