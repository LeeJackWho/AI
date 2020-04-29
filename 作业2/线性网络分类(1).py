import os
import numpy as np
import tensorflow as tf

train = np.load('train_4000.npy').astype(np.float32)
label = np.load('label.npy').astype(np.float32)

test = np.load('test_4000.npy').astype(np.float32)
test_y = np.load('true_label.npy').astype(np.float32)

# data = np.concatenate((train, test), axis=0)
# data_label = np.concatenate((label, test_y), axis=0)
#
# train = data[:int(len(data) * 0.95)]
# label = data_label[:int(len(data) * 0.95)]
#
# test = data[int(len(data) * 0.95):]
# test_y = data_label[int(len(data) * 0.95):]
# print(train.shape, label.shape, test.shape, test_y.shape)

epoch = 4000

x = tf.placeholder(shape=(None, 4000), name='x', dtype=tf.float32)
y = tf.placeholder(shape=(None, 6), name='y', dtype=tf.float32)

w1 = tf.Variable(tf.truncated_normal(shape=(4000, 2000), stddev=0.1), name='w1', dtype=tf.float32)
b1 = tf.Variable(tf.zeros(shape=(2000)), name='b1', dtype=tf.float32)
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal(shape=(2000, 1000), stddev=0.1), name='w2', dtype=tf.float32)
b2 = tf.Variable(tf.zeros(shape=(1000)), name='b2', dtype=tf.float32)
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

w3 = tf.Variable(tf.truncated_normal(shape=(1000, 500), stddev=0.1), name='w3', dtype=tf.float32)
b3 = tf.Variable(tf.zeros(shape=(500)), name='b3', dtype=tf.float32)
l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

w4 = tf.Variable(tf.truncated_normal(shape=(500, 6), stddev=0.1), name='w4', dtype=tf.float32)
b4 = tf.Variable(tf.zeros(shape=(6)), name='b4', dtype=tf.float32)
l4 = tf.matmul(l3, w4) + b4

# w5 = tf.Variable(tf.truncated_normal(shape=(100, 6), stddev=0.1), name='w5', dtype=tf.float32)
# b5 = tf.Variable(tf.zeros(shape=(6)), name='b5', dtype=tf.float32)
# l5 = tf.matmul(l4, w5) + b5

# w4 = tf.Variable(tf.truncated_normal(shape=(400, 6), stddev=0.1), name='w4', dtype=tf.float32)
# b4 = tf.Variable(tf.zeros(shape=(1, 1)), name='b', dtype=tf.float32)
# l4 = tf.matmul(l3, w4) + b4

# global_step = tf.Variable(0, trainable=False)
# variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
# variable_averages_op = variable_averages.apply(tf.trainable_variables())

# pred = tf.matmul(x, w1) + b1
pred = l4
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
regularizer = tf.contrib.layers.l2_regularizer(0.001)
loss = loss + regularizer(w1) + regularizer(w2) + regularizer(w3) + regularizer(w4)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))

# learning_rate = tf.train.exponential_decay(0.01, global_step, epoch, 0.99)
train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)
# with tf.control_dependencies([train_step, variable_averages_op]):
#     train_op = tf.no_op(name='train')

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, epoch):
        sess.run(train_step, feed_dict={x: train, y: label})
        if i % 100 == 0:
            print(
                'epoch={}, loss={}, train acc={}, test acc:{}'.format(i, sess.run(loss, feed_dict={x: train, y: label}),
                                                                      sess.run(acc, feed_dict={x: train, y: label}),
                                                                      sess.run(acc, feed_dict={x: test, y: test_y})))
            # print('test acc:{}'.format(sess.run(acc, feed_dict={x: test, y: test_y})))
    # saver.save(sess, 'Intel_model.ckpt')
    # print('model save success!!!')
