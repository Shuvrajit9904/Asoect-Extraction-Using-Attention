#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:51:16 2019

@author: shuvrajit
"""

import re
import os
import nltk




def extract_sent(path, regex):
    review_dir = []
    for file in os.listdir(path):
        with open(path+file) as f:
            txt = f.read()
            reviews = re.findall(regex, txt)
        review_dir += reviews
    
    sentence_dir = []
    for review in review_dir:
        sentences = nltk.sent_tokenize(review)
        sentence_dir += sentences
        
    return sentence_dir

#regex = r"<Content>(.*)\n<Date>" 
#path = './data/TripAdvisor/Texts/'
#
#sentence_dir = extract_sent(path, regex)