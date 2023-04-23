# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
from os import path

ELYASHIV_PATH = "C:/Users/vital_000/PycharmProjects/Deep_Learning_ass1/code/data/"
LAVI_PATH = "../data"
DATA_PATH = LAVI_PATH
TRAIN_PATH = path.join(DATA_PATH, "train")
if not path.exists(TRAIN_PATH):
    RuntimeError("train not exists")
DEV_PATH = path.join(DATA_PATH, "dev")
if not path.exists(DEV_PATH):
    RuntimeError("dev not exists")
TEST_PATH = path.join(DATA_PATH, "test")
if not path.exists(TEST_PATH):
    RuntimeError("test not exists")

def read_data(fname):
    data = []
    with open(fname, encoding="utf8") as file:
        for line in file:
            line2 = line.strip().lower().split("\t", 1)
            label, text = line.strip().lower().split("\t", 1)
            data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data(TRAIN_PATH)]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data(DEV_PATH)]
TEST  = [(text_to_bigrams(t)) for _,t in read_data(TEST_PATH)]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

## 600 most common bigrams in the training set.
TOP_K = 600
vocab = set([x for x,c in fc.most_common(TOP_K)])

## label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
## feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

# UNIGRAMS
def text_to_unigrams(text):
    return list(text)

TRAIN_UNI = [(l,text_to_unigrams(t)) for l,t in read_data(TRAIN_PATH)]
DEV_UNI   = [(l,text_to_unigrams(t)) for l,t in read_data(DEV_PATH)]
fc_uni = Counter()
for l,feats in TRAIN_UNI:
    fc_uni.update(feats)

vocab_uni = set([x for x,c in fc_uni.most_common(TOP_K)])

# label strings to IDs
L2I_UNI = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN_UNI]))))}
# feature strings (unigrams) to IDs
F2I_UNI = {f:i for i,f in enumerate(list(sorted(vocab_uni)))}