from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#importing the data
mnist=read_data_sets("data",one_hot=True)

#saving the datasets A.K.A Traning
X_traning,Y_traning=mnist.train.next_batch(5000)
X_test,Y_test=mnist.test.next_batch(200)

#placeholders for variable to be used in model
xtr=tf.placeholder(tf.float32,[None,28*28]) #traning input
ytr=tf.placeholder(tf.float32,[None,10]) #traning label
xte=tf.placeholder(tf.float32,[28*28]) #testing input

#K-near
K=3 #how many neighbors
nearest_neighbors=tf.Variable(tf.zeros([K]))

#model
distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(xtr, xte)),axis=1)) #L1
# the negitive above if so that top_k can get the lowest distance *_* its a really good hack i learned
values,indices=tf.nn.top_k(distance,k=K,sorted=False)

#a normal list to save
nn = []
for i in range(K):
    nn.append(tf.argmax(ytr[indices[i]], 0)) #taking the result indexes

#saving list in tensor variable
nearest_neighbors=nn
# this will return the unique neighbors the count will return the most common's index
y, idx, count = tf.unique_with_counts(nearest_neighbors)

pred = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]
# this is tricky count returns the number of repetation in each elements of y and then by begining from that and size begin 1
# it only returns that neighbors value : for example
# suppose a is array([11,  1,  1,  1,  2,  2,  2,  3,  3,  4,  4,  4,  4,  4,  4,  4]) so unique_with_counts of a will
#return y= (array([ 1,  2,  3,  4, 11]) count= array([3, 3, 2, 7, 1])) so argmax of count will be 3 which will be the
#index of 4 in y which is the hight number in a

#setting accuracy as 0
accuracy=0

#initialize of all variables
init=tf.global_variables_initializer()

#start of tensor session
with tf.Session() as sess:

    for i in range(X_test.shape[0]):
        #return the predicted value
        predicted_value=sess.run(pred,feed_dict={xtr:X_traning,ytr:Y_traning,xte:X_test[i,:]})

        print("Test",i,"Prediction",predicted_value,"True Class:",np.argmax(Y_test[i]))

        if predicted_value == np.argmax(Y_test[i]):
            # if the prediction is right then a double value of 1./200 is added 200 here is the number of test
                accuracy += 1. / len(X_test)
    print("Calculation completed ! ! ")
    print(K,"-th neighbors' Accuracy is:",accuracy)
