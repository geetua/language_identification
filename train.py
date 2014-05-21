#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict
import pickle
from utils import LanguageModel
from utils import prep_sentence

def train_lm(lang, files, model_file):
    """ Training language models by grabbing unigram, bigram and trigram
    statistics from train files"""
    unigrams = defaultdict(int)
    bigrams = defaultdict(int)
    trigrams = defaultdict(int)
    words = 0
    print "Training lm:", lang, " on files:", files
    for file in files:
        with codecs.open(file, 'r', 'utf-8') as f:
            for line in f.readlines():
                try:
                    index, sentence = line.strip().split('\t')
                except ValueError:
                    print "Skipping input:", line
                    pass
                sentence = prep_sentence(sentence)
                for index in range(2, len(sentence)):
                    words+=1
                    trigrams[sentence[index-2:index+1]] += 1
                    bigrams[sentence[index-1:index+1]] += 1
                    unigrams[sentence[index]] += 1
    assert(sum(unigrams.values()))==words
    lm = LanguageModel(unigrams, bigrams, trigrams)
    pickle.dump(lm, open(model_file, 'wb'))


if __name__ == '__main__':

    # Training and writing language models to file
    langs = ['deu', 'eng', 'fra']
    for lang in langs:
        train_lm(lang, ['traindata/'+lang+'.100k.txt'], 'models/'+lang+'.data')
        
