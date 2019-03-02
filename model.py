#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:35:19 2019

@author: shuvrajit
"""

import tensorflow as tf
import numpy as np

from prepare_input import *




embedding_vector_size = 300

tf.reset_default_graph()
training_inputs = tf.placeholder(shape=[None, 300, 645], 
                                         dtype=tf.float32)  

def attention(training_inputs, M):    
    
    ys = tf.expand_dims(tf.math.reduce_mean(training_inputs, axis=1), axis=1)
    d1 = tf.matmul(M, ys)
    d = tf.matmul(training_inputs, d1, transpose_a=True)    
    a = tf.nn.softmax(d)
    
    zs = tf.matmul(a, training_inputs, transpose_a=True, transpose_b=True)
    
    return zs

def reconstruct_embed(k, training_inputs, M, T):
    
    zs = attention(training_inputs[0], M)
    pt = tf.nn.softmax(tf.layers.dense(zs, k))    
    rs = tf.matmul(T, pt, transpose_b=True)
    
    return rs

def get_negative_samples(training_inputs, max_val, n):
    
    rand_idx = tf.random.uniform([n], maxval=max_val,dtype=tf.dtypes.int32)
    neg_samples = tf.gather(training_inputs, rand_idx)
    
    return neg_samples
    

def hinge_loss_neg_samples(rs, zs, n_i):
    rs = tf.squeeze(rs)
    zs = tf.squeeze(zs)
    rszs = tf.reduce_sum(tf.multiply(rs, zs))
    rsni = tf.reduce_sum(tf.multiply(tf.squeeze(rs), n_i ), 
                         1, 
                         keepdims=True )
    inner_hinge_term = tf.nn.relu(1 - rszs + rsni)
    
    return tf.reduce_sum(inner_hinge_term)

def regularization_term(T):
    
    

    
    
sample_inp = training_inputs[0]

T = tf.get_variable("T", [embedding_vector_size, k])
M = tf.get_variable("M", [embedding_vector_size, embedding_vector_size])


n_i = tf.math.reduce_mean(get_negative_samples(training_inputs, max_val, n),
                          axis =2)

rs = reconstruct_embed(k, training_inputs, M, T)
zs = attention(training_inputs[0], M)





