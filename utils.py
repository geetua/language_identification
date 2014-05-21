from math import log
import re
import sys
import unicodedata


# http://stackoverflow.com/questions/11066400/remove-punctation-from-unicode-formatted-strings                                                                                         
punc_tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                    if unicodedata.category(unichr(i)).startswith('P'))
pattern=re.compile(r'\s+')

def prep_sentence(text):
    """
    Helper method to convert sentence to desired processing format
    Text is lowercased; all unicode punctuations are removed
    I also add 2 begin and end of sentence markers to each sentence (=B & E)
    """
    text = text.lower()
    text = re.sub(pattern, ' ', text)
    text = text.translate(punc_tbl)
    text = 'B B ' + text + ' E E'
    return text

class LanguageModel():

    """
    Helper Language model class that contains the unigram, bigram and 
    trigram dictionaries of the corpus
    """

    def __init__(self, unigrams, bigrams, trigrams):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.trigrams = trigrams
