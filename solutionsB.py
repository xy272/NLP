import sys
import nltk
import math
import time
from collections import defaultdict, Counter
from collections import deque
import heapq

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        words = [START_SYMBOL] * 2
        tags = [START_SYMBOL] * 2
        for token in sentence.strip().split():
            find = token.rfind('/')
            word = token[:find]
            tag = token[find + 1:]
            words.append(word)
            tags.append(tag)
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)
        brown_words.append(words)
        brown_tags.append(tags)
    return brown_words, brown_tags

# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_tuples = []
    trigram_tuples = []
    for sentence in brown_tags:
        tokens2 = sentence
        tokens3 = [START_SYMBOL] + sentence
        bigram_tuples += list(nltk.bigrams(tokens2))
        trigram_tuples += list(nltk.trigrams(tokens3))
    # count tokens, bigrams, trigrams
    count_bigrams = Counter(bigram_tuples)
    count_trigrams = Counter(trigram_tuples)  
    count_bigrams[(START_SYMBOL, START_SYMBOL)] = len(brown_tags)
    # get log-probability
    q_values = {item: math.log(count_trigrams[item] / count_bigrams[item[0:2]], 2) for item in count_trigrams}
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    unigram_tuples = []
    for sentence in brown_words:
        unigram_tuples += sentence
    count_tokens = Counter(unigram_tuples)
    for item in count_tokens:
        if count_tokens[item] > RARE_WORD_MAX_FREQ:
            known_words.add(item)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    RARE_SYMBOL = '_RARE_'
    for sentence in brown_words:
        sentence_rare = []
        for word in sentence:
            if word not in known_words:
                sentence_rare.append(RARE_SYMBOL)
            else:
                sentence_rare.append(word)
        brown_words_rare.append(sentence_rare)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set (should not include start and stop tags)
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    words = []
    tags = []
    words_tags = []
    for sentence in brown_words_rare:
        words += sentence
    for sentence in brown_tags:
        tags += sentence
    for i in range(len(words)):
        words_tags.append(tuple([words[i],tags[i]]))
    count_words_tags = Counter(words_tags)
    count_tags = Counter(tags)
    for key, p in count_words_tags.items():
        e_values[key] = math.log(float(p) / count_tags[key[1]], 2)
    taglist = set(count_tags)
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    sorted(emissions)
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    bp = {}
    pi = {}
    for dev_tokens in brown_dev_words:
        bp = {}
        pi = {}
        L = len(dev_tokens)
        tokens = [w if w in known_words else RARE_SYMBOL for w in dev_tokens]   
        possible_taglist = []
        for token in tokens:
            tmp = [key[1] for key in e_values.keys() if token == key[0]]
            if len(tmp) > 0:
                possible_taglist.append(tmp)
            else:
                possible_taglist.append(taglist)

        # First Word
        for w in possible_taglist[0]:
            word_tag = (tokens[0], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            pi[(0, START_SYMBOL, w)] = q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
            bp[(0, START_SYMBOL, w)] = START_SYMBOL
        if L == 1:
            max_prob = float('-Inf')
            v_max = None
            # finding the max probability of tag
            for v in possible_taglist[0]:
                score = pi.get((0, START_SYMBOL, v), LOG_PROB_OF_ZERO) + q_values.get((START_SYMBOL, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
                if score > max_prob:
                    max_prob = score
                    v_max = v
            tags = []
            tags.append(v_max)
        else:
            # Second Word
            for w in possible_taglist[0]:
                for u in possible_taglist[1]:
                    word_tag = (tokens[1], u)
                    trigram = (START_SYMBOL, w, u)
                    pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
                    bp[(1, w, u)] = START_SYMBOL
            if L > 2:
                # Following word
                for k in range(2, L):
                    for u in possible_taglist[k-1]:
                        for v in possible_taglist[k]:
                            word_tag = (tokens[k], v)
                            max_prob = float('-Inf')
                            max_tag = None
                            for w in possible_taglist[k-2]: 
                                trigram = (w, u, v)
                                score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
                                if (score > max_prob):
                                    max_prob = score
                                    max_tag = w
                            bp[(k, u, v)] = max_tag
                            pi[(k, u, v)] = max_prob
            # finding the max of last two tags
            max_prob = float('-Inf')
            v_max, u_max = None, None
            # finding the max probability of last two tags
            for u in possible_taglist[L-2]:
                for v in possible_taglist[L-1]:
                    score = pi.get((L-1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
                    if score > max_prob:
                        max_prob = score
                        u_max = u
                        v_max = v
            # append tags in reverse order
            tags = []
            tags.append(v_max)
            tags.append(u_max)

            for count, k in enumerate(range(L-3, -1, -1)):
                tags.append(bp[(k+2, tags[count+1], tags[count])])
            # reverse tags
            tags.reverse()
        tagged_sentence = []
        # stringify tags paired with word without start and stop symbols
        for k in range(0, L):
            tagged_sentence += [dev_tokens[k], "/", str(tags[k]), " "]
        tagged_sentence.append('\n')
        tagged.append(''.join(tagged_sentence))

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in range(len(brown_words)) ]
    training = [list(x) for x in training]
    # IMPLEMENT THE REST OF THE FUNCTION HERE
    t0 = nltk.tag.DefaultTagger('NOUN')
    t1 = nltk.tag.BigramTagger(training, backoff=t0)
    t2 = nltk.tag.TrigramTagger(training, backoff=t1)
    tagged = []
    for sentence in brown_dev_words:
        sentence_tag = t2.tag(sentence)
        tagged.append(' '.join([word + '/' + tag for word, tag in sentence_tag]) + '\n')
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = '/home/classes/cs477/data/' # absolute path to use the shared data
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print(f"Part B time: {str(time.clock())} sec")

if __name__ == "__main__": main()
