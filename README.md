# CIFAR 10 in Python

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.  
There are 50000 training images and 10000 test images. 

[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)  

**Download**  

    wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

* [CIFAR 10 Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

**Extract**  

    tar -xvzf cifar-10-python.tar.gz

**Files:**

* `load_cifar.py`

load `data_batch_1` data (images) and labels (classes) into a np.array  
[Batch to Array](https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546)

* `imsave.py`

save an image from CIFAR-10 as JPG

* `classe_lables.py` 

load `batches.meta` label_names (classes) and print them  

* `filter_class.py` 

write class label index from `data_batch_1` to class_label.txt

**Classes:**  

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