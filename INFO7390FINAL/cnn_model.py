#from: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
import tensorflow as tf
import numpy as np
import re
import itertools
import csv
import os
from pathlib import Path
from numpy import array
from collections import Counter

#load 5 classes of data
def load_data_and_labels5(data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x=[]
    y=[]
    # Load data from files
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            file = data_path+file
            with open(file) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    x.append(row[0])
                    if row[1] == 'bbcnews':
                        y.append([0,0,0,0,1])
                    elif row[1] == 'linkedin':
                        y.append([0,0,0,1,0])
                    elif row[1] == 'NASA':
                        y.append([0,0,1,0,0])
                    elif row[1] == 'nytimes':
                        y.append([0,1,0,0,0])
                    elif row[1] == 'steam':
                        y.append([1,0,0,0,0])					
    y = array (y)
    return [x, y]

#load 3 classes of data	
def load_data_and_labels3(data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    x=[]
    y=[]
    # Load data from files
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            file = data_path+file
            with open(file) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    if row[1] =='bbcnews' or row[1] =='linkedin' or row[1] =='NASA' :
                        x.append(row[0])
                        if row[1] == 'bbcnews':
                            y.append([0,0,1])
                        elif row[1] == 'linkedin':
                            y.append([0,1,0])
                        elif row[1] == 'NASA':
                            y.append([1,0,0])				
    y = array (y)
    return [x, y]	

#method to separate for each batch	
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


			
#The CNN model we are using form http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
#This model by DENNY BRITZ is licensed under the Apache License Version 2.0 https://www.apache.org/licenses/LICENSE-2.0
class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
	  #Change statement: we add few parameters to change the model config
      embedding_size, filter_sizes, num_filters,act_function, cost_function, init, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
				#Change statement: we add act_function to switch between relu and elu
                if act_function==0:
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                elif act_function==1 :
                    h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")					
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
				#Change statement: we add uniform to switch between random and uniform
                initializer=tf.contrib.layers.xavier_initializer(uniform=init))
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
		    #Change statement: we add cost_function to switch between softmax_cross_entropy_with_logits and sigmoid_cross_entropy_with_logits
            if cost_function == 0:
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            elif cost_function == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
