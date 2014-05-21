import codecs
from collections import defaultdict
from math import log
import optparse
import pickle
import sys
from utils import prep_sentence

def get_ngrams(sentence):
    """ Returns iterator for unigram, bigram, trigram in sentence

    Args:
        sentence = (str) sentence to be split

    Returns:
        (str) tuple of unigram, bigram, trigram
    """
    sentence = prep_sentence(sentence)
    for index in range(2, len(sentence)):
        trigram = sentence[index-2:index+1]
        bigram = sentence[index-1:index+1]
        unigram = sentence[index]
        yield unigram, bigram, trigram


def compute_loglikelihood(sentence, ngrams, lms):
    """ Return loglikelihood, lang pairs for given sentence
    Args:
         sentence = (str) sentence to operate on
         ngrams = (int) ngram statistic to use
         lms = (dict) language model keyed by lang 
    
    Returns:
         (lst) (log-likelihood, lang) tuples sorted in ascending 
               order of log-likelihood
    """
    loglikelihoods = defaultdict(float)
    lmcounts = defaultdict(int)
    for unigram, bigram, trigram in get_ngrams(sentence):                
        for lang, lm in lms.iteritems():
            if ngrams>3 or ngrams<1: 
                raise AttributeError('Incorrect value of ngrams passed')
            # Note: I check to see if ngrams counts exist in lm model and 
            # simply ignore uknown ngrams statistics. Works well in this case 
            # but in case of sparser data, I would have implemented a smoothing 
            # (ex: Leave One Out or Good Turing) or tried some backoff technique 
            if ngrams==3 and trigram in lm.trigrams and bigram in lm.bigrams:
                bip = lm.trigrams[trigram]/float(lm.bigrams[bigram])
            elif ngrams==2 and bigram in lm.bigrams and unigram in lm.unigrams:
                bip = lm.bigrams[bigram]/float(lm.unigrams[unigram])
            elif ngrams==1 and unigram in lm.unigrams:
                bip = lm.unigrams[unigram]/float(sum(lm.unigrams.values()))
            loglikelihoods[lang] += -log(bip)
            lmcounts[lang] += 1
    result = [(loglikelihoods[lang]/lmcounts[lang], lang) for lang in langs]
    result.sort()
    return result

def compute_statistics_test_corpus(ngrams, lms):
    """ Computes accuracy of language prediction over test corpus
    using precomputed language models
    
    Args:
         ngrams = (int) ngram statistic to use
         lms = (dict) language model keyed by lang 
    """
    correct = 0
    total = 0 
    for test_lang in langs:
        testfile = 'testdata/'+test_lang+'.test.txt' 
        with codecs.open(testfile, 'r', 'utf-8') as f:
            for line in f.readlines():
                index, sentence = line.strip().split('\t')
                result = compute_loglikelihood(sentence, ngrams, lms)
                # picking lang with lowest negative average log probability
                (score, predicted) = result[0]
                #print testfile, index, test_lang, predicted
                if test_lang==predicted: correct+=1
                total+=1
    print "Accuracy on lm using ngrams(", ngrams, ") : ", float(correct)/total*100


if __name__ == '__main__':

    # loading trained lms
    langs = ['deu', 'eng', 'fra']
    lms = {}
    for lang in langs:
        lm = pickle.load(open('models/'+lang+'.data', 'rb'))
        lms[lang] = lm
    
    ''' Sanity check: Setting sanity check to True results in 
    running the language identification over pre trained language models
    and computing accuracy using unigram, bigram and trigram character lms
    '''
    sanity_check = True
    if sanity_check:
        for ngrams in range(1,4):
            compute_statistics_test_corpus(ngrams, lms)
    print '\n\n'

    parser = optparse.OptionParser()
    parser.add_option('-n', type="int", dest="ngrams")
    parser.add_option('-i', type="string", dest="input")
    (opts, args) = parser.parse_args()

    if not opts.ngrams or not opts.input:
        sys.exit(0)

    ngrams = opts.ngrams
    sentence = opts.input.decode('ISO-8859-15')
    result = compute_loglikelihood(sentence, ngrams, lms)
    # picking lang with lowest negative average log probability
    print "Predictions:", result
    (score, predicted) = result[0]
    print "Input sentence is identified as: ", predicted

    
    


        
