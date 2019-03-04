#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:14:25 2019

@author: shuvrajit
"""

import numpy as np
import gensim
import operator
import pickle

from functools import reduce
from build_sentence_corp import extract_sent
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

def least_freq_words(k, corp):
    corp = reduce(operator.concat, corp)
    corp = Counter(corp)
    infreq_words = set()
    for key in corp:
        if corp[key] < k:
            infreq_words.add(key)
    return infreq_words

def stop_infreq_words(sentence_dir):
    
    vocab_count = {}
    for sent in sentence_dir:
        for word in sent.split():
            if word.lower() in vocab_count:
                vocab_count[word.lower()] += 1
            else:
                vocab_count[word.lower()] = 1
        
    infreq_words = set()
    k = 10    
    for key, val in vocab_count.items():
        if val < k:
            infreq_words.add(key)
    
    for word in stopwords.words('english'):
        infreq_words.add(word)
    
    return infreq_words
        
def processed_sent(sentence_dir):
    sent_processed_dir = []
    max_sent_length = 0
    elim_words = stop_infreq_words(sentence_dir)
    
    for sent in sentence_dir:    
        tokenizer = RegexpTokenizer(r'\w+')
        sent_tokens = tokenizer.tokenize(sent)
        sent_tokens = [w.lower() for w in sent_tokens if not w.lower() in elim_words]
        sent_processed_dir.append(sent_tokens)
        max_sent_length = max(max_sent_length, len(sent_tokens))
    
    return max_sent_length, sent_processed_dir



def prepare_inp():
    text_path = './data/TripAdvisor/Texts/'
    regex = r"<Content>(.*)\n<Date>" 
    sentence_dir = extract_sent(text_path, regex)
    
#    embedding_vector_size = 300
#    embedding_path = 'data/GoogleNews-vectors-negative300.bin'
#    embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, 
#                                 binary=True)
    
    max_sent_length, sent_processed_dir = processed_sent(sentence_dir)
    
    return max_sent_length, sent_processed_dir

max_sent_length, sent_processed_dir = prepare_inp()
#test_sent_processed_dir = sent_processed_dir[:100000]

embedding_vector_size = 300
embedding_path = 'data/GoogleNews-vectors-negative300.bin'
embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, 
                             binary=True)

est_max_sent_length = 100
sentence_embedding = []
unk = set()

ct = 0
batch_size = 50

for sent in sent_processed_dir:
    ct += 1
    sent_embed_init = np.zeros((est_max_sent_length, embedding_vector_size))
    if len(sent) < est_max_sent_length:
        for i,word in enumerate(sent):
            try:
                sent_embed_init[i] = embedding[word]
            except:
                unk.add(word)
                pass
    
        sentence_embedding.append(sent_embed_init.T)
    if ct % 50 == 0 or ct == len(sent_processed_dir)-1:
        idx = ct // 50
        filename = './data/batch_embedding/batch-'+str(idx)+'.dat'
        sentence_embedding = np.asarray(sentence_embedding, dtype=np.float32)
        with open(filename, 'wb') as f:            
            pickle.dump(sentence_embedding, f)
        sentence_embedding = []
        
        
        
        
    
        
    
