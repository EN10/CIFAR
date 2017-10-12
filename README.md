# CIFAR 10 in Python

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.  
There are 50000 training images and 10000 test images. 

[CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)  

**Download**  

    wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

* [CIFAR 10 Python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

**Extract**  

    tar -xvzf cifar-10-python.tar.gz

[Batch to Array](https://gist.github.com/juliensimon/273bef4c5b4490c687b2f92ee721b546)

**Files:**

* `cifar_array.py`

load `data_batch_1` data (images) and labels (classes) into a np.array  

* `classes.py` 

load `batches.meta` label_names (classes) and print them  

* `cats.py` 

write cats index from `data_batch_1` to cats.txt

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