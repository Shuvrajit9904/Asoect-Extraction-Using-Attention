#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:35:19 2019

@author: shuvrajit
"""

import tensorflow as tf
#import numpy as np

#from prepare_input import *




#training_inputs = tf.placeholder(shape=[None, 300, 100], 
#                                         dtype=tf.float32)  

def attention(training_inputs, M):    
    
    ys = tf.expand_dims(tf.reduce_mean(training_inputs, axis=2), axis=2)
    ET = tf.transpose(training_inputs, perm=(0,2,1))
    d1 = tf.tensordot(ET, M, axes=1)
    d = tf.matmul(d1, ys)
    a = tf.nn.softmax(d)
    
    zs = tf.matmul(training_inputs, a)
    
    return tf.transpose(zs, perm = (0,2,1))

def reconstruct_embed(k, zs, training_inputs, M, T):
    
    pt = tf.nn.softmax(tf.layers.dense(zs, k))
    rs = tf.tensordot(pt, tf.transpose(T), axes=1)
    
    return rs

def get_negative_samples(training_inputs, max_val, n):
    
    rand_idx = tf.random_uniform([n], maxval=max_val, dtype=tf.int32)
    neg_samples = tf.gather(training_inputs, rand_idx)
    
    return neg_samples
    

def hinge_loss_neg_samples(rs, zs, n_i):

    rszs = tf.reduce_sum(tf.multiply(rs, zs))
    rsni = tf.reduce_sum(tf.multiply(rs, n_i ), 
                         1, 
                         keepdims=True )
    inner_hinge_term = tf.nn.relu(1 - rszs + rsni)
    
    return tf.reduce_sum(inner_hinge_term)

def regularization_term(T):
    
    tensor_norms = tf.norm(T, axis=1, keepdims=True)
    Tn = tf.div(T, tensor_norms)    
    num_rows = tf.shape(T)[0]    
    U = tf.norm(tf.matmul(Tn, Tn, transpose_b=True) - tf.eye(num_rows))
    
    return U
    
def total_loss(rs, zs, T, training_inputs, max_val, n):
    n_i = tf.reduce_mean(get_negative_samples(training_inputs, 
                                                   max_val, n),
                          axis =2)

    hinge_loss =  hinge_loss_neg_samples(rs, zs, n_i)  
    U = regularization_term(T)
    
    total_loss = U + hinge_loss
    
    return total_loss
    
    






