import cPickle
import numpy as np

def load():
    path = 'cifar-10-batches-py/'
    file = 'data_batch_1'

    f = open(path+file, 'rb')
    dict = cPickle.load(f)
    images = dict['data']
    #images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = np.array(images)   #   (10000, 3072)
    labelarray = np.array(labels)   #   (10000,)
    
    return imagearray, labelarray

#load()