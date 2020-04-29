import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import warnings
warnings.filterwarnings('ignore')
mnist=input_data.read_data_sets("./MNIST_data",one_hot=True)
train_X,train_Y,test_X,test_Y=mnist.train.images,mnist.train.labels,\
mnist.test.images,mnist.test.labels
print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
