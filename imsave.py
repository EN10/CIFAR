import cPickle
import numpy as np

path = 'cifar-10-batches-py/'
db1 = 'data_batch_1'

f = open(path+db1, 'rb')
dict = cPickle.load(f)
images = dict['data']
images = np.reshape(images, (10000, 3, 32, 32))

from scipy.misc import imsave
imsave('image.jpg', images[46])     # image no 46