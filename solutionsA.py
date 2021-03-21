from collections import deque, defaultdict, Counter
import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    
    # initialization
    unigram_tuples = []
    bigram_tuples = []
    trigram_tuples = []
    
    # process each sentence
    for sentence in training_corpus:      
        tokens0 = sentence.strip().split() # contain '.' with word?
        # tokens0 = nltk.word_tokenize(sentence)
        tokens1 = tokens0 + [STOP_SYMBOL]
        tokens2 = [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]

        unigram_tuples += tokens1
        bigram_tuples += list(nltk.bigrams(tokens2))
        trigram_tuples += list(nltk.trigrams(tokens3))

    # count tokens, bigrams, trigrams
    count_tokens = Counter(unigram_tuples)
    count_bigrams = Counter(bigram_tuples)
    count_trigrams = Counter(trigram_tuples)
    # get total count
    len_tokens = sum(count_tokens.values())
    
    count_tokens[START_SYMBOL] = len(training_corpus)
    count_bigrams[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    
    # get log-probability
    unigram_p = {tuple([item]): math.log(count_tokens[item] / len_tokens, 2) for item in count_tokens}
    bigram_p = {item: math.log(count_bigrams[item] / count_tokens[item[0]], 2) for item in count_bigrams}
    trigram_p = {item: math.log(count_trigrams[item] / count_bigrams[item[0:2]], 2) for item in count_trigrams}
    
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    #output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    sorted(unigrams_keys)
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    sorted(bigrams_keys)
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    sorted(trigrams_keys)
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        sent_score = 0
        # tokens0 = nltk.word_tokenize(sentence)
        tokens0 = sentence.strip().split()
        if n == 1:
            tokens = tokens0 + [STOP_SYMBOL]
            tokens = [tuple([item]) for item in tokens]
        elif n == 2:
            tokens2 = [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
            tokens = list(nltk.bigrams(tokens2))
        elif n == 3:
            tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
            tokens = list(nltk.trigrams(tokens3))
        else:
            raise ValueError('The value of parameter is invalid.')
        for token in tokens:
            try:
                p = ngram_p[token]
            except KeyError:
                p = MINUS_INFINITY_SENTENCE_LOG_PROB
            sent_score += p
        scores.append(sent_score)
    return scores
    
    
# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lambda1 = 1.0/3
    for sentence in corpus:
        inter_score = 0
        # tokens0 = nltk.word_tokenize(sentence)
        tokens0 = sentence.strip().split()
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        trigram_tuples = list(nltk.trigrams(tokens3))
        for trigram_key in trigram_tuples:
            try:
                tri_p = trigrams[trigram_key]
            except KeyError:
                tri_p = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                bri_p = bigrams[trigram_key[1:]]
            except KeyError:
                bri_p = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                uni_p = unigrams[trigram_key[2:]]
            except KeyError:
                uni_p = MINUS_INFINITY_SENTENCE_LOG_PROB
            inter_score += math.log(lambda1 * (2 ** tri_p) + lambda1 * (2 ** bri_p) + lambda1 * (2 ** uni_p), 2)
        scores.append(inter_score)
    return scores

# TODO: IMPLEMENT THIS FUNCTION
# As above, but with modified lambda values
import numpy as np
def linearscore_newlambdas(unigrams, bigrams, trigrams, corpus):
    scores = []
    tri_p = []
    bri_p = []
    uni_p = []
    for sentence in corpus:
        inter_score = 0
        tokens0 = sentence.strip().split()
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        trigram_tuples = list(nltk.trigrams(tokens3))
        for trigram_key in trigram_tuples:
            try:
                tri_p.append(2**trigrams[trigram_key])
            except KeyError:
                tri_p.append(2**MINUS_INFINITY_SENTENCE_LOG_PROB)
            try:
                bri_p.append(2**bigrams[trigram_key[1:]])
            except KeyError:
                bri_p.append(2**MINUS_INFINITY_SENTENCE_LOG_PROB)
            try:
                uni_p.append(2**unigrams[trigram_key[2:]])
            except KeyError:
                uni_p.append(2**MINUS_INFINITY_SENTENCE_LOG_PROB)
    tri_p = np.array(tri_p)
    bri_p = np.array(bri_p)
    uni_p = np.array(uni_p)
    
    # gradient descent to find best lambda
    lambda1 = 1.0/3
    lambda2 = 1.0/3
    lambda3 = 1.0/3
    step = 0.001
    log_likelihood = []
    for iteration in range(100):
        sum_p = lambda1 * uni_p + lambda2 * bri_p + (1 - lambda1 - lambda2) * tri_p
        lambda1 += step * np.mean((uni_p-tri_p) / sum_p)
        lambda2 += step * np.mean((bri_p-tri_p) / sum_p)
        lambda3 = 1 - lambda1 - lambda2
        log_likelihood.append(np.mean(np.log2(lambda1 * uni_p + lambda2 * bri_p + lambda3 * tri_p)))
    
    # calculate scores for each sentence
    start = 0
    for sentence in corpus:
        tokens0 = sentence.strip().split()
        M = len(tokens0) + 1
        score0 = np.log2(lambda1 * uni_p[start:start+M] + lambda2 * bri_p[start:start+M] + lambda3 * tri_p[start:start+M])
        scores.append(np.sum(score0))
        start += M
    
    return scores

DATA_PATH = '/home/classes/cs477/data/' # absolute path to the shared data
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)
    linearscores_modlambda = linearscore_newlambdas(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.reg.txt')
    score_output(linearscores_modlambda, OUTPUT_PATH + 'A3.newlambdas.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print(f"Part A time: {str(time.clock())} sec")

if __name__ == "__main__": main()



