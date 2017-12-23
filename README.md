# CIFAR 10 in Python

The CIFAR-10 dataset consists of 60000 32x32 colour images in [10 classes](https://github.com/EN10/CIFAR#classes), with 6000 images per class.  
There are 50000 training images and 10000 test images.  
The dataset is divided into five training batches and one test batch, each with 10000 images.

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  

### Download

    wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

* [CIFAR 10 Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

### Extract

    tar -xvzf cifar-10-python.tar.gz

### Files:

* `sigmoid.py` : [Based on DL.ai cats.py](https://github.com/EN10/DL.ai/blob/master/w2/cats.py)

* `5_layer.py:` : [Based on DL.ai 5_layer_model.py](https://github.com/EN10/DL.ai/blob/master/w4/5_layer_model.py)

* `load_cifar.py` : load `data_batch_1` data (images) and labels (classes) into a np.array 

* `imsave.py` : save an image from CIFAR-10 as JPG

* `class_labels.py` : load `batches.meta` label_names (classes) and print them  

* `filter_class.py` : write class label index from `data_batch_1` to class_label.txt

### Accuracy
`load_cifar.py` uses `random.seed(1)` for consistant accuracy  

`sigmoid.py:`

    train accuracy: 89.0 %
    test accuracy: 72.0 %

`5_layer.py:`

    train accuracy: 99.0 %
    test accuracy: 70.0 %

### Classes:

0 : airplane  
1 : automobile  
2 : bird  
3 : cat  
4 : deer  
5 : dog  
6 : frog  
7 : horse  
8 : ship  
9 : truck  

### Ref:

[Batch to Array Code](https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546)

### Compatability
[imsave.py](https://github.com/EN10/CIFAR/blob/master/imsave.py) uses `from scipy.misc import imsave` which is deprecated  
this requires `sudo pip install scipy==0.16.1`  
[save.py](https://github.com/EN10/KerasCIFAR/blob/master/save.py) PIL version