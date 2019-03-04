#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:36:33 2019

@author: shuvrajit
"""

from prepare_input import *
from build_sentence_corp import *
from model import *
import pickle
import numpy as np
import tensorflow as tf


f = open("./paralab/Aspect-Extraction-Using-Attention/data/T_matrix.pickle",
         'rb')
_T = pickle.load(f)
f.close()

with open('./paralab/Aspect-Extraction-Using-Attention/data/sent_embed.dat', 'wb') as f:
    sentence_embedding = np.asarray(sentence_embedding)
    pickle.dump(sentence_embedding, f)
    
with open('./data/sample_sent_embed', 'rb') as f:
    sent_embedding_sample = pickle.load(f)
    

sent_embedding_sample = sent_embedding_sample
sent_embedding_sample = np.asarray(sent_embedding_sample)

embedding_vector_size = 300
k = 14
max_val = len(sent_embedding_sample)
n = 5

tf.reset_default_graph()
training_inputs = tf.placeholder(shape=[None, 300, 100], dtype=tf.float32)  

g = tf.Graph()
T = tf.transpose(tf.cast(tf.get_variable("T", initializer=_T), 
                         dtype=tf.float32))
M = tf.get_variable("M", 
                    [embedding_vector_size, embedding_vector_size], 
                    dtype=tf.float32)

zs = attention(training_inputs, M)
rs = reconstruct_embed(k, zs, training_inputs, M, T)
loss = total_loss(rs, zs, T, training_inputs, max_val, n)

train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

variables_names = [v.name for v in tf.trainable_variables()]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
    M2 = sess.run(tf.trainable_variables())
    for i in range(100):
        print("Epoch: ",i)
        ls, _ = sess.run([loss, train_op], 
                         feed_dict={training_inputs:sent_embedding_sample})
        print("loss", ls)
    
    