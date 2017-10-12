from load_cifar import load
import numpy as np

images , _ = load()
images = np.reshape(images, (10000, 3, 32, 32))

from scipy.misc import imsave
imsave('cat.jpg', images[26])     # image no #