#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:14:25 2019

@author: shuvrajit
"""

import numpy as np
import gensim
import operator

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
        if corp[key] < 10:
            infreq_words.add(key)
    return infreq_words

def stop_infreq_words(sentence_dir):
    
    vocab_count = {}
    for sent in sentence_dir:
        for word in sent.split():
            if word in vocab_count:
                vocab_count[word] += 1
            else:
                vocab_count[word] = 1
        
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
        sent_tokens = [w for w in sent_tokens if not w in elim_words]
        sent_processed_dir.append(sent_tokens)
        max_sent_length = max(max_sent_length, len(sent_tokens))
    
    return max_sent_length, sent_processed_dir



text_path = './data/TripAdvisor/Texts/'
regex = r"<Content>(.*)\n<Date>" 
sentence_dir = extract_sent(text_path, regex)

embedding_vector_size = 300
embedding_path = 'data/GoogleNews-vectors-negative300.bin'
embedding = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, 
                             binary=True)

max_sent_length, sent_processed_dir = processed_sent(sentence_dir)

sentence_embedding = []
#for sent in sent_processed_dir:
#    sent_embed_init = np.zeros((max_sent_length, embedding_vector_size))
#    for i,word in enumerate(sent):
#        try:
#            sent_embed_init[i] = embedding[word]
#        except:
#            pass
#    
#    sentence_embedding.append(sent_embed_init.T)
##        
##    
##        
##    