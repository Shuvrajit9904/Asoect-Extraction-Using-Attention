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
    
    return tf.nn.softmax(d)

M = tf.get_variable("M", [embedding_vector_size, embedding_vector_size])