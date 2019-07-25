#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:04:53 2019

@author: suk
"""

import gzip
import simplejson

def parse(filename):
  f = gzip.open(filename, 'r')
  entry = {}
  for l in f:
    l = l.strip()
    colonPos = l.find(b':')
    if colonPos == -1:
      yield entry
      entry = {}
      continue
    eName = l[:colonPos]
    rest = l[colonPos+2:]
    entry[eName] = rest
  yield entry

for e in parse("/home/suk/Desktop/mls/news_classification/text_classification_using_gans_hans/data/reviews_Health_and_Personal_Care_5.json.gz"):
  print(simplejson.dumps(e))