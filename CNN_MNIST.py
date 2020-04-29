####导入包(time 包用于计时)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.examples.tutorials.mnist import input_data

###########提取手写字体数据集
mnist=input_data.read_data_sets("./MNIST_data",one_hot=True)
train_X,train_Y,test_X,test_Y=mnist.train.images,mnist.train.labels,\
mnist.test.images,mnist.test.labels

######输入
n_classes=10 ##手写字体10个类
x=tf.placeholder(tf.float32,[None,28*28]) #输入数据为28*28的图像，None表示对训练样本个数没有要求
y=tf.placeholder(tf.float32,[None,n_classes]) #输出数据为10个类别，None表示对训练样本个数没有要求
keep_prob = tf.placeholder(tf.float32) #用于dropout

epochs=1 #网络训练次数
learning_rate=0.01 #学习率
batch_size=200 #训练块的大小(mini-batch SGD)
batch_num=int(mnist.train.num_examples/batch_size) #块的大小
dropout=0.75 #dropout的概率
filter_width=5 #滤波器的宽度
filter_height=5 #滤波器的高度
depth_in=1 #输入数据通道数
depth_out1=64 #隐含层1的通道数(特征数目)
depth_out2=128#隐含层2的通道数(特征数目)
f_height=28 #输入图像尺寸

######ops:Weights and bias
Weights={"wc1":tf.Variable(tf.random_normal([filter_height,filter_width,depth_in,depth_out1])),\
        "wc2":tf.Variable(tf.random_normal([filter_height,filter_width,depth_out1,depth_out2])),\
        "wd1":tf.Variable(tf.random_normal([int((f_height*f_height/16)*depth_out2),1024])),\
        "out":tf.Variable(tf.random_normal([1024,n_classes]))}

bias={"bc1":tf.Variable(tf.random_normal([depth_out1])),\
      "bc2":tf.Variable(tf.random_normal([depth_out2])),\
      "bd1":tf.Variable(tf.random_normal([1024])),\
      "out":tf.Variable(tf.random_normal([n_classes]))}

##############卷积层和最大池化层定义

def conv2d(x,W,b,stride=1):
    x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding="SAME")
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x)
def maxpool2d(x,stride=2):
    return tf.nn.max_pool(x,ksize=[1,stride,stride,1],strides=[1,stride,stride,1],padding="SAME")


####构建卷积神经网络+全连接层
def conv_net(x_,W,b,dropout):
    x=tf.reshape(x_,[-1,28,28,1])
    ####卷积层 1######
    conv1=conv2d(x,W["wc1"],b["bc1"])
    conv1=maxpool2d(conv1,2)
    
    ####卷积层 2######
    conv2=conv2d(conv1,W["wc2"],b["bc2"])
    conv2=maxpool2d(conv2,2)
    ####卷积层与全连接层连接#####
    fc1=tf.reshape(conv2,[-1,W["wd1"].get_shape().as_list()[0]])
    ####全连接层#####
    fc1=tf.matmul(fc1,W["wd1"])
    fc1=tf.add(fc1,b["bd1"])
    fc1=tf.nn.relu(fc1)
    
    ###### dropout用于防止过拟合####
    fc1=tf.nn.dropout(fc1,dropout)
    
    ######输出层####
    out=tf.matmul(fc1,W["out"])
    out=tf.add(out,b["out"])
    
    return out

########定义loss、optimizer#####
pred=conv_net(x,Weights,bias,keep_prob) #网络预测值
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y)) #损失函数
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost) #优化器

####模型评价
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
####################################################
## 训练
####################################################
start_time = time.time()
with tf.Session() as sess:
    sess.run(init)
    for  i  in range(epochs):
        for  j  in range(batch_num):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_x,y:batch_y,keep_prob:0.75})
            loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob: 1.})
            if epochs % 1 == 0:
                print("Epoch:", '%04d' % (i+1),"cost=", "{:.9f}".format(loss),"Training accuracy","{:.5f}".format(acc))
    print('Optimization Completed')
    end_time = time.time()
    print('Total processing time:',end_time - start_time)
    y1 = sess.run(pred,feed_dict={x:mnist.test.images[:256],keep_prob: 1})
    test_classes = np.argmax(y1,1)
    print('Testing Accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob: 1}))
  
