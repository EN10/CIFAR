import cPickle
import numpy as np

path = 'cifar-10-batches-py/'
db1 = 'data_batch_1'

f = open(path+db1, 'rb')
dict = cPickle.load(f)
images = dict['data']
images = np.reshape(images, (10000, 3, 32, 32))
imagearray = np.array(images)

i = 32
img = np.dstack((images[i][0], images[i][1], images[i][2]))

from scipy.misc import imsave
imsave('image.jpg', img)