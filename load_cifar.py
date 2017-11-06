import cPickle
import numpy as np
import random
random.seed(1) # set a seed so that the results are consistent

def load_batch():
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

def create_datasets(imagearray, labelarray):
    train_set_x = np.empty((200,3072))
    train_set_y = np.empty((1,200),dtype=np.int16)

    i = 0
    j = 0
    while (j < 200):                #   200 train images
        x = random.randint(0,1)
        if (labelarray[i] == 3):    #   Cats
            train_set_x[j] = imagearray[i]
            train_set_y[0,j] = 1    #   Cat is True
            j+=1
        elif (x % 2 == 0 and labelarray[i] != 3):   #    NOT Cats
            train_set_x[j] = imagearray[i]
            train_set_y[0,j] = 0    # Cat is False
            j+=1
        i+=1
        
    train_set_x = train_set_x.T     #   Reshape to (3072, 200) 
    
    test_set_x = np.empty((50,3072))                #   50 test images
    test_set_y = np.empty((1,50),dtype=np.int16)

    i = 0
    j = 0
    while (j < 50):
        x = random.randint(0,1)
        if (labelarray[9999-i] == 3):#  In Reverse Order is Cat
            test_set_x[j] = imagearray[9999-i]
            test_set_y[0,j] = 1
            j+=1
        elif (x % 2 == 0 and labelarray[i] != 3):
            test_set_x[j] = imagearray[9999-i]
            test_set_y[0,j] = 0
            j+=1
        i+=1

    test_set_x = test_set_x.T       #   Reshape to (3072, 50)

    train_set_x = train_set_x/255.  #   0-255 -> 0-1
    test_set_x = test_set_x/255.

    return train_set_x, train_set_y, test_set_x, test_set_y

#load()