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
    
    
    

text_path = './data/TripAdvisor/Texts/'
regex = r"<Content>(.*)\n<Date>" 
sentence_dir = extract_sent(text_path, regex)

embedding_vector_size = 300
embedding_path = 'data/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, 
                             binary=True)



sent_processed_dir = []
max_sent_length = 0
for sent in sentence_dir:    
    tokenizer = RegexpTokenizer(r'\w+')
    sent_tokens = tokenizer.tokenize(sent)
    sent_tokens = [w for w in sent_tokens if not w in stopwords.words('english')]
    sent_processed_dir.append(sent_tokens)
    max_sent_length = max(max_sent_length, len(sent_tokens))
    

sent_embed_init = np.zeros((embedding_vector_size, max_sent_length))