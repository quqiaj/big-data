# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv

origin_f = open('/home/lee/assignment3/test.csv', 'rb')
new_f = open('/home/lee/assignment3/test1.csv', 'wb+')
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
       writer.writerow(row)
origin_f.close()
new_f.close()

test_data=np.zeros([400,56],dtype=float)

test=csv.reader(open('/home/lee/assignment3/test1.csv'))
j=-1
for line1 in test:
    temp=line1[1]
    j=j+1
    for i in range(len(temp)):
        if temp[i]=='A':
            test_data[j,i*4]=1
        if temp[i]=='C':
            test_data[j,i*4+1]=1
        if temp[i]=='G':
            test_data[j,i*4+2]=1
        if temp[i]=='T':
            test_data[j,i*4+3]=1
#print test_data

origin_f = open('/home/lee/assignment3/train.csv', 'rb')
new_f = open('/home/lee/assignment3/train1.csv', 'wb+')
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
       writer.writerow(row)
origin_f.close()
new_f.close()
train_data=np.zeros([2000,56],dtype=float)

train=csv.reader(open('/home/lee/assignment3/train1.csv'))
j=-1
for line1 in train:
    temp=line1[1]
    j=j+1
    for i in range(len(temp)):

        if temp[i]=='A':
            train_data[j,i*4]=1
        if temp[i]=='C':
            train_data[j,i*4+1]=1
        if temp[i]=='G':
            train_data[j,i*4+2]=1
        if temp[i]=='T':
            train_data[j,i*4+3]=1

#print train_data

train_label=np.zeros([2000,2],dtype=float)
train=csv.reader(open('/home/lee/assignment3/train1.csv'))
k=-1
for line2 in train:
    k=k+1
    temp=line2[2]
    if temp=='0':
        train_label[k,0]=1
    if temp=='1':
        train_label[k,1]=1
#print train_label




INPUT_NODE=56
OUTPUT_NODE=2


LAYER1_NODE=10
LAYER2_NODE=5


LEARNING_RATE_BASE=0.6
LEARNING_RATE_DECAY=0.99
REGULARIZATION_RATE=0.0001
TRAINING_STEPS=4000
MOVING_AVERAGE_DECAY=0.99


train=np.zeros([2000,58])
train[:,:56]=train_data[:,:]
train[:,56:58]=train_label[:,:]

def add_layer(input_tensor,avg_class,w1,b1,w2,b2,w3,b3):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,w1)+b1)
        layer2=tf.nn.relu(tf.matmul(layer1,w2)+b2)
        return tf.matmul(layer2,w3)+b3
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(w1))+avg_class.average(b1))
        layer2=tf.nn.relu(tf.matmul(layer1,avg_class.average(w2))+avg_class.average(b2))
        return tf.matmul(layer2,avg_class.average(w3))+avg_class.average(b3)



x=tf.placeholder(tf.float32,shape=[None,INPUT_NODE],name='x-input')
y_=tf.placeholder(tf.float32,shape=[None,OUTPUT_NODE],name='y-input')


w1=tf.Variable(tf.truncated_normal(shape=[INPUT_NODE,LAYER1_NODE],stddev=0.1))
b1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

w2=tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE,LAYER2_NODE],stddev=0.1))
b2=tf.Variable(tf.constant(0.1,shape=[LAYER2_NODE]))

w3=tf.Variable(tf.truncated_normal(shape=[LAYER2_NODE,OUTPUT_NODE],stddev=0.1))
b3=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))


y=add_layer(x,None,w1,b1,w2,b2,w3,b3)


global_step=tf.Variable(0,trainable=False)


variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

variable_averages_op=variable_averages.apply(tf.trainable_variables())


average_y=add_layer(x,variable_averages,w1,b1,w2,b2,w3,b3)


cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
cross_entrip_mean=tf.reduce_mean(cross_entropy)


regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
regularization=regularizer(w1)+regularizer(w2)+regularizer(w3)
loss=cross_entrip_mean+regularization


learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,900,LEARNING_RATE_DECAY)
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
train_op=tf.group(train_step,variable_averages_op)


correct_prediction=tf.equal(tf.arg_max(average_y,1),tf.arg_max(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver=tf.train.Saver()

test_feed={x:test_data}
with tf.Session() as sess:
    saver.restore(sess,'/home/lee/assignment3/saver/model.ckpt')
    test_result = sess.run(tf.arg_max(average_y, 1), feed_dict=test_feed)
    #print test_result
    with open('/home/lee/assignment3/sampleSubmission4.csv','w') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['id','prediction'])
        for i in range(len(test_result)):
            if test_result[i]==0:
                writer.writerow([i,0])
            if test_result[i]==1:
                writer.writerow([i,1])
    with open('/home/lee/assignment3/sampleSubmission4.csv','r') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            print line