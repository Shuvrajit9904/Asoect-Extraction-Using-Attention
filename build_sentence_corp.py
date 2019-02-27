#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 21:51:16 2019

@author: shuvrajit
"""

import re
import os

path = './data/TripAdvisor/Texts/'


p = re.compile(r"<Content>.*\n<Date>")

review_dir = []
for file in os.listdir(path):
    with open(path+file) as f:
        txt = f.read()
        reviews = re.findall(r"<Content>(.*)\n<Date>", txt)
    review_dir += reviews
