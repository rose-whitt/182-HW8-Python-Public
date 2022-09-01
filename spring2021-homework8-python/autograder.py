# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 8, Problem 2

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from provided.py, but they have
# to be copied over here.

import math
import random
import numpy
from collections import *

#################   PASTE PROVIDED CODE HERE AS NEEDED   #################

def read_pos_file(filename):
    """
    Parses an input tagged text file.
    Input:
    filename --- the file to parse
    Returns:
    The file represented as a list of tuples, where each tuple
    is of the form (word, POS-tag).
    A list of unique words found in the file.
    A list of unique POS tags found in the file.
    """
    file_representation = []
    unique_words = set()
    unique_tags = set()
    f = open(str(filename), "r")
    for line in f:
        if len(line) < 2 or len(line.split("/")) != 2:
            continue
        word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
        tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
        file_representation.append( (word, tag) )
        unique_words.add(word)
        unique_tags.add(tag)
    f.close()
    return file_representation, unique_words, unique_tags

class HMM:
    """
    Simple class to represent a Hidden Markov Model.
    """
    def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
        self.order = order
        self.initial_distribution = initial_distribution
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix


##########################   HELPER FUNCTIONS   ##########################

def count_sentences(training_data: list):
    """
    Input: 
        training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
    
    Output:
        count: an int that represents the number of sentences in the input

    """

    count = 0
    for element in training_data:
        if element[1] == '.' or element[1] == '?' or element[1] == '!':
            count += 1
    return count

def parse_sentences(training_data: list):
    """
    Input:
        training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
    
    Output:
        sentences: a list of lists where each element is a sentence and each element of the sentence
                    is the (word, tag)
    """

    sentences = []
    temp_sent = []

    for element in training_data:
        # print(element)
        if element[1] != '.':
            # print(element)
            temp_sent.append(element)
        if element[1] == '.':
            temp_sent.append(element)
            sentences.append(temp_sent)
            temp_sent = []
            
    return sentences

def first_words(sentences: list) :
    """
    Input:
        sentences: a list given by parse_sentences

    Output:
        first_words: a list where each element is the first word of each sentence and each element is
                    formatted (word, tag)
    """

    first_words = []
    for sent in sentences:
        first_words.append(sent[0])
    
    return first_words

def first_two(sentences: list):
    """
    Input:
        sentences: a list given by parse_sentences

    Output:
        two_words: a list where each element is a list with two elements representing the first two
                    words of each sentence with each word being formmated as (word, tag)
    """

    two_words = []
    for idx, sent in enumerate(sentences):
        if len(sent) >= 2:
            two_words.append([sent[0], sent[1]])
    
    return two_words

def unique_words_test(x: list):
    words = []
    for i in x:
        if i[0] not in words:
            words.append(i[0])
    return words

def unique_tags_test(x: list):
    tags = []
    for i in x:
        if i[1] not in tags:
            tags.append(i[1])
    return tags

def print_initial_prob(init_dist):
    """
    Input:
        init_dist: 2d default dict
    """
    x = []
    for second_tag, val1 in init_dist.items():
        for first_tag, val in val1.items():
            x.append(str((second_tag, first_tag)) + str(": ") + str(val))
            #print(str((second_tag, first_tag)) + str(": ") + str(val))
    return x

def print_emission_matrix(emi_mat):
    """
    Input:
        emi_mat: 2d default dict
    """
    #x = []
    for tag, val1 in emi_mat.items():
        for word, val in val1.items():
            if val != 0:
                print(str((tag, word)) + str(": ") + str(val))

def print_transition_matrix(trans_mat):
    """
    Input:
        trans_mat: 3d default dict
    """
    #x = []
    for tag2, val2 in trans_mat.items():
        for tag1, val1 in val2.items():
            for tag, val in val1.items():
                print(str((tag2, tag1, tag)) + str(": ") + str(val))
                #x.append(str((tag2, tag1, tag)) + str(": ") + str(val))
    

#####################  TEST DATA  #####################
# # TEST 0 AND 1
# test0_list = [('hw7', 'N'), ('is', 'V'), ('difficult', 'A'), ('.', '.')]

# # TEST A
# # first sentence
# testa_list = read_pos_file('testdata_tagged.txt')[0][0:26]
# unique_words_a = unique_words_test(testa_list)


# # TEST B
# # first two sentences
# testb_list = read_pos_file('testdata_tagged.txt')[0][0:63]
# sentence_b = []
# for x in testb_list:
#     sentence_b.append(x[0])
# #print(sentence_b)
# print(len(sentence_b))
# unique_words_b = unique_words_test(testb_list)
# unique_tags_b = unique_tags_test(testb_list)

# # TEST C
# # first three sentences
# testc_list = read_pos_file('testdata_tagged.txt')[0][0:92]
# sentence_c = []
# for x in testc_list:
#     sentence_c.append(x[0])
# #print(sentence_c)

# # TEST ALL
# test_all_list = read_pos_file('testdata_tagged.txt')[0]
# #print(test_all_list)
# sentence_all = []
# for x in test_all_list:
#     sentence_all.append(x[0])
# #print(sentence_all)
# unique_words_all = read_pos_file('testdata_tagged.txt')[1]
# unique_tags_all = read_pos_file('testdata_tagged.txt')[2]

# read_training = read_pos_file('training.txt')
# training_list = read_training[0]
# unique_words_training = read_training[1]
# unique_tags_training = read_training[2]
#####################  STUDENT CODE BELOW THIS LINE  #####################

def compute_counts(training_data: list, order: int) -> tuple:
    """
    Inputs:
        training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
        order: an int that represents the order of the HMM
    Outputs:
        - a tuple containing the number of tokens in training_data
        - c_ti_wi: a dictionary that contains that contains 𝐶(𝑡𝑖,𝑤𝑖) for every unique tag and unique word (keys correspond to tags)
            - 𝐶(𝑡𝑖,𝑤𝑖) is the number of times word i is tagged with tag i
        - c_ti: a dictionary that contains 𝐶(𝑡𝑖) as above
            - 𝐶(𝑡𝑖) is the number of times tag i appears
        - c_ti1_ti: a dictionary that contains 𝐶(𝑡𝑖−1,𝑡𝑖) as above
            - 𝐶(𝑡𝑖−1,𝑡𝑖) is the number of times the tag sequence 𝑡𝑖−1, 𝑡𝑖 appears
            
            and, if order is 3:
        - c_ti2_ti1_ti: dictionary that contains 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖)
            - 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖) is the number of times the tag sequence 𝑡𝑖−2, 𝑡𝑖−1, 𝑡𝑖
    """
    # INITIALIZATION
    num_tokens = len(training_data)
    c_ti_wi = defaultdict(lambda: defaultdict(int))
    c_ti = defaultdict(int)
    c_ti1_ti = defaultdict(lambda: defaultdict(int))
    if order == 3:
        c_ti2_ti1_ti = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # COUNTING
    for idx in range(0, num_tokens):
        word = training_data[idx][0]
        tag = training_data[idx][1]

        c_ti_wi[tag][word] += 1

        c_ti[tag] += 1

        if idx >= 1:
            tag_behind = training_data[idx - 1][1]

            c_ti1_ti[tag_behind][tag] += 1
        
        if order == 3:
            if idx >= 2:
                tag_behind = training_data[idx - 1][1]
                tag_behind2 = training_data[idx - 2][1]
                c_ti2_ti1_ti[tag_behind2][tag_behind][tag] += 1

    if order == 2:
        return (num_tokens, c_ti_wi, c_ti, c_ti1_ti)
    else:
        return (num_tokens, c_ti_wi, c_ti, c_ti1_ti, c_ti2_ti1_ti)

def compute_initial_distribution(training_data: list, order: int) -> dict:
    """
    Inputs:
        training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
        order: an int that represents the order of the HMM
    Outputs:
        if order is 2:
            - init_dist_copy: a dictionary that represents the probability that tag ti appears at the beginning of a sentence
        if order is 3:
            - init_dist_copy: a dictionary that represents the probability that tag bigram ti, tj appears at the beginning of a sentence
    """
    
    num_sentences = count_sentences(training_data)
    # print(num_sentences)
    all_sentences = parse_sentences(training_data)
    # print(all_sentences)
    
    if order == 2:
        f_w = first_words(all_sentences)
        #print(f_w)

        #num sentences that start with each
        #init_dist = {}
        init_dist = defaultdict(int)
        for word in f_w:
            #word[1] is the tag
            if word[1] not in init_dist:
                init_dist[word[1]] = 1
            else:
                init_dist[word[1]] += 1

        #init_dist_copy = {}
        init_dist_copy = defaultdict(int)

        for key, val in init_dist.items():
            init_dist_copy[key] = val/num_sentences

        return init_dist_copy
    elif order == 3:
        s_w = first_two(all_sentences)
        #print(s_w)

        #init_dist = {}
        init_dist = defaultdict(lambda: defaultdict(int))

        for pair in s_w:
            first_tag = pair[0][1]
            second_tag = pair[1][1]

            if first_tag not in init_dist:
                #init_dist[first_tag] = {second_tag: 1}
                init_dist[first_tag][second_tag] = 1
            #if first tag in init_dist
            else:
                #if both are in
                if second_tag in init_dist[first_tag]:
                    #up count
                    init_dist[first_tag][second_tag] += 1
                #first in, second not
                else:
                    #init_dist[first_tag] = {second_tag: 1}
                    init_dist[first_tag][second_tag] = 1

        init_dist_copy = defaultdict(lambda: defaultdict(int))
        for key, val in init_dist.items():
            #print(key, val)
            for key1, val1 in val.items():
                #print(key1, val1)
                #init_dist_copy[key] = {key1 : val1/num_sentences}
                init_dist_copy[key][key1] = val1/num_sentences

        return init_dist_copy

def compute_emission_probabilities(unique_words: list, unique_tags: list, W: dict, C: dict) -> dict:
    """
    Inputs:
        - unique_words: set returned by read_pos_file that contains unique words
        - unique_tags: set returned by read_pos_file that contains unique tags
        - W: a dictionary computed by compute_counts that represents 𝐶(𝑡𝑖,𝑤𝑖)
            - 𝐶(𝑡𝑖,𝑤𝑖) is the number of times word i is tagged with tag i
        - C: a dictionary computed by compute_counts that represents 𝐶(𝑡𝑖)
            - 𝐶(𝑡𝑖) is the number of times tag i appears
    Outputs:
        - a dictionary that represents the emission matrix where the keys are the tags
    """
    #print(W)
    #print(C)
    #emi_prob = {}
    emi_prob = defaultdict(lambda: defaultdict(int))

    for item in unique_tags:
        for item2 in unique_words:
            if C[item] != 0:
                emi_prob[item][item2] = (W[item][item2])/C[item]
            else:
                emi_prob[item][item2] = 0
    return emi_prob

def compute_lambdas(unique_tags: list, num_tokens: int, C1: dict, C2: dict, C3: dict, order: int) -> list:
    """
    Inputs:
        - unique_tags: set returned by read_pos_file that contains unique tags
        - num_tokens: number of words (length of training data)
        - C1: a dictionary computed by compute_counts that represents 𝐶(𝑡𝑖)
            - 𝐶(𝑡𝑖) is the number of times tag i appears
        - C2: a dictionary that contains 𝐶(𝑡𝑖−1,𝑡𝑖)
            - 𝐶(𝑡𝑖−1,𝑡𝑖) is the number of times the tag sequence 𝑡𝑖−1, 𝑡𝑖 appears
            - {ti-1: {ti: num}}
        - C3: dictionary that contains 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖)
            - 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖) is the number of times the tag sequence 𝑡𝑖−2, 𝑡𝑖−1, 𝑡𝑖
            - {ti-2: {ti-1: {ti: num}}}
        - order: an int that represents the order of the HMM
    Outputs:
        tuple of (lambda_0, lambda_1, lambda_2)
    """
    lambda_0 = 0
    lambda_1 = 0
    lambda_2 = 0

    if order == 2:
        # iterate over 𝐶(𝑡𝑖−1,𝑡𝑖)
        #tag_1 is ti-1
        for tag_1, tag_1_vals in C2.items():
            # tag is ti
            # tag_vals is 𝐶(𝑡𝑖−1,𝑡𝑖)
            for tag, tag_vals in tag_1_vals.items():
                if tag_vals > 0:
                    #print(tag_1, tag)

                    if num_tokens != 0:
                        a0 = (C1[tag] - 1) / num_tokens
                        #print("a0: " + str(a0))
                    else:
                        a0 = 0
                    
                    if (C1[tag_1] - 1) != 0:
                        a1 = (C2[tag_1][tag] - 1) / (C1[tag_1] - 1)
                        #print("a1: " + str(a1))
                    else:
                        a1 = 0

                    # argmax
                    temp_max = max(a0, a1)
                    if a0 == a1:
                        i = 0
                        lambda_0 += tag_vals
                    else:
                        if temp_max == a0:
                            i = 0
                            lambda_0 += tag_vals
                        elif temp_max == a1:
                            lambda_1 += tag_vals

                    # print("lambda 0: " + str(lambda_0))
                    # print("lambda 1: " + str(lambda_1))
                    # print("lambda 2: " + str(lambda_2))
                    # print("")
        
        sum_lambdas = lambda_0 + lambda_1
        lambda_0 /= sum_lambdas
        lambda_1 /= sum_lambdas
        #print("sum lambdas: " + str(lambda_0 + lambda_1 + lambda_2))
        return [lambda_0, lambda_1, 0.0]



    elif order == 3:
        # iterate over 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖)
        for tag_2, tag_2_vals in C3.items():
            #tag_2 is ti-2
            # print("------------")
            # print(tag_2, tag_2_vals)
            for tag_1, tag_1_vals in tag_2_vals.items():
                #tag_1 is ti-1
                # print(tag_1, tag_1_vals)
                # print("")
                for tag, tag_vals in tag_1_vals.items():
                    # tag is ti
                    # tag_vals is 𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖)
                    if tag_vals > 0:
                        # print(tag, tag_vals)
                        # print("")
                        #print(tag_2, tag_1, tag)

                        if num_tokens != 0:
                            #print(C1[tag])
                            a0 = (C1[tag] - 1) / num_tokens
                            #lambda_0 += a0
                            #print("a0: " + str(a0))
                            #print("lambda 0: " + str(lambda_0))
                            
                        else:
                            a0 = 0
                        
                        if (C1[tag_1] - 1) != 0:
                            a1 = (C2[tag_1][tag] - 1) / (C1[tag_1] - 1)
                            #print("a1: " + str(a1))
                            #print("")
                        else:
                            a1 = 0
                            #print("")
                        
                        if (C2[tag_2][tag_1] - 1) != 0:
                            a2 = (tag_vals - 1) / (C2[tag_2][tag_1] - 1)
                            #print("a2: " + str(a2))
                        else:
                            a2 = 0
                        
                        #argmax
                        temp_max = max(a0, a1, a2)
                        if a0 == a1 == a2:
                            i = 0
                            lambda_0 += tag_vals
                        else:
                            if temp_max == a0:
                                i = 0
                                lambda_0 += tag_vals
                            elif temp_max == a1:
                                i = 1
                                lambda_1 += tag_vals
                            elif temp_max == a2:
                                i = 2
                                lambda_2 += tag_vals
                        # print("lambda 0: " + str(lambda_0))
                        # print("lambda 1: " + str(lambda_1))
                        # print("lambda 2: " + str(lambda_2))
                        # print("")

        sum_lambdas = lambda_0 + lambda_1 + lambda_2	
        lambda_0 /= sum_lambdas
        lambda_1 /= sum_lambdas
        lambda_2 /= sum_lambdas
        #print("sum lambdas: " + str(lambda_0 + lambda_1 + lambda_2))
        return [lambda_0, lambda_1, lambda_2]

                    
    return 0

def transition_probs(training_data: list, unique_tags: list, num_tokens: int, order: int, use_smoothing: bool):
    # need
    #	lambas
    # 	compute counts
    comp_counts = compute_counts(training_data, order)
    num_tokens = comp_counts[0]
    c_ti_wi = comp_counts[1]
    C1 = comp_counts[2]
    C2 = comp_counts[3]
    #print(C2)
    print("")
    if order == 3:
        C3 = comp_counts[4]
    else:
        C3 = 0
    
    lambdas = compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order)
    if order == 2:
        
        if use_smoothing:
            lambda_0 = lambdas[0]
            lambda_1 = lambdas[1]
            # print("lambda 0: " + str(lambda_0))
            # print("lambda 1: " + str(lambda_1))
        else:
            lambda_0 = 0
            lambda_1 = 1
            lambda_2 = 0
            # print("lambda 0: " + str(lambda_0))
            # print("lambda 1: " + str(lambda_1))
            # print("lambda 2: " + str(lambda_2))

        bigram_transition_matrix = defaultdict(lambda: defaultdict(int))

        #ti
        for tag1 in unique_tags:
            #ti-1
            for tag2 in unique_tags:
                if C1[tag2] != 0 and num_tokens != 0:
                    bigram_transition_matrix[tag2][tag1] = lambda_1 * (C2[tag2][tag1] / C1[tag2]) + lambda_0 * (C1[tag1] / num_tokens)
                elif C1[tag2] == 0 and num_tokens != 0:
                    bigram_transition_matrix[tag2][tag1] = lambda_0 * (C1[tag1] / num_tokens)
                else:
                    bigram_transition_matrix[tag2][tag1] = 0
        return bigram_transition_matrix

    elif order == 3:
        if use_smoothing:
            lambda_0 = lambdas[0]
            lambda_1 = lambdas[1]
            lambda_2 = lambdas[2]
            # print("lambda 0: " + str(lambda_0))
            # print("lambda 1: " + str(lambda_1))
            # print("lambda 2: " + str(lambda_2))
        else:
            lambda_0 = 0
            lambda_1 = 0
            lambda_2 = 1
            # print("lambda 0: " + str(lambda_0))
            # print("lambda 1: " + str(lambda_1))
            # print("lambda 2: " + str(lambda_2))

        trigram_transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for tag1 in unique_tags:
            #ti-1
            for tag2 in unique_tags:
                for tag3 in unique_tags:
                    if C2[tag3][tag2] != 0 and C1[tag2] != 0 and num_tokens != 0:
                        trigram_transition_matrix[tag3][tag2][tag1] = lambda_2 * (C3[tag3][tag2][tag1] / C2[tag3][tag2]) + lambda_1 * (C2[tag2][tag1] / C1[tag2]) + lambda_0 * (C1[tag1] / num_tokens)
                    elif C2[tag3][tag2] == 0 and C1[tag2] != 0 and num_tokens != 0:
                        trigram_transition_matrix[tag3][tag2][tag1] = lambda_1 * (C2[tag2][tag1] / C1[tag2]) + lambda_0 * (C1[tag1] / num_tokens)
                    elif C2[tag3][tag2] == 0 and C1[tag2] == 0 and num_tokens != 0:
                        trigram_transition_matrix[tag3][tag2][tag1] = lambda_0 * (C1[tag1] / num_tokens)
                    else:
                        trigram_transition_matrix[tag3][tag2][tag1] = 0
        return trigram_transition_matrix

def build_hmm(training_data: list, unique_tags: list, unique_words: list, order: int, use_smoothing: bool):
    """
    Inputs:
        - training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
        - unique_tags: set returned by read_pos_file that contains unique tags
        - unique_words: set returned by read_pos_file that contains unique words
        - order: an int that represents the order of the HMM
        - use_smoothing: boolean value
    Outputs:
        A: transition matrix
        E: emission probability matrix
                       'N'   'V'   'A'    '.'
        'hw7':       [     ,     ,     ,      ]
        'is':        [     ,     ,     ,      ]
        'difficult': [     ,     ,     ,      ]
        '.':         [     ,     ,     ,      ]

        {'N': {'hw7': 1.0}}
        prob_dist: probability distribution on the initial states

    Write function build_hmm(training_data, unique_tags, unique_words, order, use_smoothing)
    where training_data is the same as that input parameter to compute_counts and use_smoothing
    is a Boolean parameter. The function returns an (fully trained) HMM.
    If use_smoothing is True, the function uses the 𝜆s as computed by compute_lambdas;
    otherwise, it uses 𝜆0=𝜆2=0 and 𝜆1=1 in the case of a bigram mode,
    and 𝜆0=𝜆1=0 and 𝜆2=1 in the case of a trigram model.
    """

    comp_counts = compute_counts(training_data, order)
    num_tokens = comp_counts[0]
    c_ti_wi = comp_counts[1]
    c_ti = comp_counts[2]
    c_ti1_ti = comp_counts[3]
    if order == 3:
        c_ti2_ti1_ti = comp_counts[4]
    else:
        c_ti2_ti1_ti = 0
    initial_distribution = compute_initial_distribution(training_data, order)
    emission_matrix = compute_emission_probabilities(unique_words, unique_tags, c_ti_wi, c_ti)
    transition_matrix = transition_probs(training_data, unique_tags, num_tokens, order, use_smoothing)
    #print(transition_matrix)
    #print("poo")
    
    trained_hmm = HMM(order, initial_distribution, emission_matrix, transition_matrix)
    return trained_hmm

def update_hmm(hmm, sentence: list):
    """
    Inputs:
        - hmm: a hidden markov model made from the training.txt data
        - sentence: a list of words made from the testdata_untagged.txt data
    Outputs:
        - dict: the new emission matrix
    """
    # print("1")
    EPSILON = 0.00001
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    # print("2")
    for word in sentence:
        for tag in unique_tags:
            hmm.emission_matrix[tag][word] += EPSILON
    # print("3")
    for tag in unique_tags:
        prob_sum = sum(hmm.emission_matrix[tag].values())
        for word in hmm.emission_matrix[tag].keys():
            hmm.emission_matrix[tag][word] /= prob_sum
    # print("4")
 
def trigram_viterbi(hmm, sentence: list) -> list:

    #print(print_initial_prob(hmm.initial_distribution))
    #emi_mat_print = hmm.emission_matrix
    #print_transition_matrix(hmm.transition_matrix)
    

    # INITIALIZATION
    viterbi = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    backpointer = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))

    for prev in unique_tags:
        for prev_prev in unique_tags:
            if hmm.initial_distribution[prev_prev][prev] != 0 and hmm.emission_matrix[prev_prev][sentence[0]] != 0 and hmm.emission_matrix[prev][sentence[1]] != 0:
                viterbi[prev_prev][prev][0] = math.log(hmm.initial_distribution[prev_prev][prev]) + math.log(hmm.emission_matrix[prev_prev][sentence[0]]) + math.log(hmm.emission_matrix[prev][sentence[1]])
            else:
                viterbi[prev_prev][prev][0] = -1 * float('inf')
    
    # DYNAMIC PROGRAMMING

    # len(sentence) - 1 is important
    for t in range(1, len(sentence) - 1):

        backpointer["No_Path"]["No_Path"][t] = "No_Path"

        # current node
        for cur in unique_tags:
            for l_prime in unique_tags:

                #max
                max_value = -1 * float('inf')
                #argmax
                max_state = None

                # x', l', cur
                for x_prime in unique_tags:

                    # val1 is v[x', l', i - 1]
                    # which is the prob of l' being 1 behind cur and x' being two behind
                    val1 = viterbi[x_prime][l_prime][t - 1]

                    # val2 is A[x', l', cur]
                    val2 = -1 * float('inf')
                    
                    if hmm.transition_matrix[x_prime][l_prime][cur] != 0:
                        val2 = math.log(hmm.transition_matrix[x_prime][l_prime][cur])
                    
                    cur_val = val1 + val2
                    if cur_val > max_value:
                        max_value = cur_val
                        max_state = x_prime
            
                val3 = -1 * float('inf')
                if hmm.emission_matrix[cur][sentence[t + 1]] != 0:
                    val3 = math.log(hmm.emission_matrix[cur][sentence[t + 1]])
                
                viterbi[l_prime][cur][t] = max_value + val3
                if max_state == None:
                    backpointer[l_prime][cur][t] = "No_Path"
                else:
                    backpointer[l_prime][cur][t] = max_state

    # Termination
    max_value = -1 * float('inf')
    last_state = None
    second_to_last_state = None
    final_time = len(sentence) - 2
    for l_prime in unique_tags:
        for x_prime in unique_tags:
            # print(viterbi[s_prime][final_time])
            # print("")
            if viterbi[x_prime][l_prime][final_time] > max_value:
                max_value = viterbi[x_prime][l_prime][final_time]
                last_state = l_prime
                second_to_last_state = x_prime
    
    if last_state == None:
        last_state = "No_Path"
    if second_to_last_state == None:
        second_to_last_state = "No_Path"

    #print("second to last: " + str(second_to_last_state))
    # Traceback
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence) - 1], last_state))
    tagged_sentence.append((sentence[len(sentence) - 2], second_to_last_state))

    for i in range(len(sentence) - 3, -1, -1):
        next_tag = tagged_sentence[-1][1]
        next_next_tag = tagged_sentence[-2][1]
        curr_tag = backpointer[next_tag][next_next_tag][i + 1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    
    # print("viterbi: " + str(viterbi))
    # print("")
    # print("backpointer: " + str(backpointer))
    return tagged_sentence


# # test 7
# order7 = 3
# initial_distribution_7 = {'Coin1': {'Coin1': 0.25, 'Coin2': 0.25}, 'Coin2': {'Coin1': 0.25, 'Coin2': 0.25}}
# emission_matrix_7 = {'Coin1': {'Heads': 0.9, 'Tails': 0.1}, 'Coin2': {'Heads': 0.5, 'Tails': 0.5}}
# transition_matrix_7 = {'Coin1': {'Coin1': {'Coin1': 0.5, 'Coin2': 0.5}, 'Coin2': {'Coin1': 0.5, 'Coin2': 0.5}}, 'Coin2': {'Coin1': {'Coin1': 0.5, 'Coin2': 0.5}, 'Coin2': {'Coin1': 0.5, 'Coin2': 0.5}}}
# hmm7 = HMM(2, initial_distribution_7, emission_matrix_7, transition_matrix_7)
# expected_output = [('Heads', 'Coin1'), ('Heads', 'Coin1'), ('Tails', 'Coin2')]
# input_sentence7 = ['Heads', 'Heads', 'Tails']

# #trigram_viterbi_test1(hmm7, input_sentence7)

# #test 7 but for bigram
# transition_matrix_7_bigram = {'Coin1': {'Coin1': 0.5, 'Coin2': 0.5}, 'Coin2': {'Coin1': 0.5, 'Coin2': 0.5}}
# initial_distribution_7_bigram = {'Coin1': 0.25, 'Coin2': 0.75}
# #hmm7_bigram = HMM(2, initial_distribution_7_bigram, emission_matrix_7, transition_matrix_7_bigram)


# # TEST
# read_training = read_pos_file('training.txt')
# training_list = read_training[0]
# unique_words_training = read_training[1]
# unique_tags_training = read_training[2]

# #SET UP TRAINING HMM
# # print("before build hmm")
# # hmm_all_trigram = build_hmm(training_list, unique_tags_training, unique_words_training, 3, True)
# # print("before update hmm")
# # update_hmm(hmm_all_trigram, sentence_all)

# # print("before build hmm")
# # hmm_test_tagged = build_hmm(test_all_list, unique_tags_all, unique_words_all, 3, True)
# # print("before update hmm")
# # update_hmm(hmm_test_tagged, sentence_b)

# # FIRST TWO SENTENCES
# # print("before trigram")
# # tagged_sentence_b = trigram_viterbi(hmm_all_trigram, sentence_all)
# # print("")
# # print(tagged_sentence_b)

# # keep upping
# # testb_plus_list = read_pos_file('testdata_tagged.txt')[0][0:63]
# # sentence_b_plus = []
# # for x in testb_plus_list:
# #     sentence_b_plus.append(x[0])


# # FIRST THREE SENTENCES
# # fucks up
# # print("before trigram")
# # tagged_sentence_c = trigram_viterbi(hmm_all_trigram, sentence_c)
# # print(tagged_sentence_c)

# # ALL SENTENCES
# # print("before trigram")
# # tagged_sentence_all = trigram_viterbi(hmm_all_trigram, sentence_all)
# # print(tagged_sentence_all)
# # print("hi")

def test_trigram(tagged_sentence: list, test_list, training_list):
    count = 0
    for idx, pair in enumerate(tagged_sentence):
        if pair not in training_list and pair not in test_list:
            if count < 10:
                print("WRONG AT: " + str(pair))
                print("INDEX OF PAIR: " + str(idx))
                print("")
            count += 1

#print(test_trigram(tagged_sentence_b, training_list, testb_list))


#####################  TEST FUNCTION  #####################
def bigram_viterbi(hmm, sentence):
    """
    Run the Viterbi algorithm to tag a sentence assuming a bigram HMM model.
    Inputs:
      hmm --- the HMM to use to predict the POS of the words in the sentence.
      sentence ---  a list of words.
    Returns:
      A list of tuples where each tuple contains a word in the
      sentence and its predicted corresponding POS.
    """

    # Initialization
    viterbi = defaultdict(lambda: defaultdict(int))
    
    backpointer = defaultdict(lambda: defaultdict(int))
    
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    
    for tag in unique_tags:
        #print("tag: " + str(tag))
        #print(hmm.initial_distribution[tag])
        # print(tag, hmm.initial_distribution[tag], hmm.emission_matrix[tag][sentence[0]])
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            #print(tag)
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')
    # print("viterbi: " + str(viterbi))
    # print("")
    # Dynamic programming.
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        # print(t)
        for s in unique_tags:
            #max
            max_value = -1 * float('inf')
            #argmax
            max_state = None
            for s_prime in unique_tags:
                val1= viterbi[s_prime][t-1]
                val2 = -1 * float('inf')
                if hmm.transition_matrix[s_prime][s] != 0:
                    val2 = math.log(hmm.transition_matrix[s_prime][s])
                curr_value = val1 + val2
                if curr_value > max_value:
                    max_value = curr_value
                    max_state = s_prime
            val3 = -1 * float('inf')
            if hmm.emission_matrix[s][sentence[t]] != 0:
                val3 = math.log(hmm.emission_matrix[s][sentence[t]])
            viterbi[s][t] = max_value + val3
            if max_state == None:
                backpointer[s][t] = "No_Path"
            else:
                backpointer[s][t] = max_state
    for ut in unique_tags:
        string = ""
        for i in range(0, len(sentence)):
            if (viterbi[ut][i] != float("-inf")):
                string += str(int(viterbi[ut][i])) + "\t"
            else:
                string += str(viterbi[ut][i]) + "\t"
                
    #print(viterbi)
    # Termination
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        # print(viterbi[s_prime][final_time])
        # print("")
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"
    
    print("viterbi: " + str(viterbi))
    print("backpointer: " + str(backpointer))
    print("last state: " + str(last_state))
    # Traceback
    # tagged_sentence: Z
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    # for i <- L - 2 downto 0 do
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    
    print("tagged sentence: " + str(tagged_sentence))
    return tagged_sentence

# #print(bigram_viterbi(hmm7_bigram, input_sentence7))
# sentence1a = ['The', 'New', 'Deal', 'was', 'a', 'series', 'of', 'domestic', 'programs', 'enacted', 'in', 'the', 'United', 'States', 'between', '1933', 'and', '1936', ',', 'and', 'a', 'few', 'that', 'came', 'later', '.']
# #print(sentence1a)

# #print(bigram_viterbi(hmm_all, sentence_all))
# #print(bigram_viterbi(hmma1, sentence1a))
# #print(bigram_viterbi(hmm7, input_sentence7))



# #hmm_all = build_hmm(training_list, unique_tags_training, unique_words_training, 2, True)
# #before_emi = hmm_all.emission_matrix.copy()
# #print(before_emi)
# #update_hmm(hmm_all, sentence_all)
# #after_emi = hmm_all.emission_matrix
# #print(before_emi == after_emi)

# #print(hmm_all.emission_matrix)
# #print(bigram_viterbi(hmm_all, sentence_all))
# #hmm_training = build_hmm(training_list, unique_tags_training, unique_words_training, 2, True)

# viterbi_test7_bigram = {'Coin2': {0: -0.9808292530117262, 1: -2.367123614131617, 2: -3.1656313103493887}, 'Coin1': {0: -1.491654876777717, 1: -1.779336949229498, 2: -4.775069222783489}}
# backpointer_test7_bigram = {'No_Path': {1: 'No_Path', 2: 'No_Path'}, 'Coin2': {1: 'Coin2', 2: 'Coin1'}, 'Coin1': {1: 'Coin2', 2: 'Coin1'}}
def english_viterbi(v: dict, bp: dict, order: int):
    """
    Input:
        v: a defaultdict of the viterbi dict
        order: int representing bigram (2, first order) or trigram(3, second order)

    """
    if order == 2:
        for state, vals in v.items():
            for position, probability in vals.items():
                print("The max probability that the state " + str(state) + " will be at position " + str(position) + " of the sentence is " + str(probability) + ".")
        
        for prev_state, vals in bp.items():
            for position, cur_state in vals.items():
                print("Given that the state before me was " + str(prev_state) + " and I am currently at position " + str(position) + ", the best path would be if my position had the state " + str(cur_state) + ".")
    else:
        print("poo")
    
#english_viterbi(viterbi_test7_bigram, backpointer_test7_bigram, 2)

def test_compute_counts(test):
    """
    Input:
        - test: a string representing which test we are doing
    """
    # first sentence
    if test == 'a':
        counts = test_a_compute_counts
        #print(counts[4])
        num_tokens = 26
        c_ti_wi = {
                'DT': {'The': 1, 'a': 2, 'the': 1},
                'NNP': {'New': 1, 'Deal': 1, 'United': 1},
                'VBD': {'was': 1, 'came': 1},
                'NN': {'series': 1},
                'IN': {'of': 1, 'in': 1, 'between': 1},
                'JJ': {'domestic': 1, 'few': 1},
                'NNS': {'programs': 1},
                'VBN': {'enacted': 1},
                'NNPS': {'States': 1},
                'CD': {'1933': 1, '1936': 1},
                'CC': {'and': 2},
                ',': {',': 1},
                'WDT': {'that': 1},
                'RB': {'later': 1},
                '.': {'.': 1}
                }
        c_ti = {
            'DT': 4,
            'NNP': 3,
            'VBD': 2,
            'NN': 1,
            'IN': 3,
            'JJ': 2,
            'NNS': 1,
            'VBN': 1,
            'NNPS': 1,
            'CD': 2,
            'CC': 2,
            ',': 1,
            'WDT': 1,
            'RB': 1,
            '.': 1
            }
        c_ti1_ti = {
            'DT': {'NNP': 2, 'NN': 1, 'JJ': 1},
            'NNP': {'NNP': 1, 'VBD': 1, 'NNPS': 1},
            'VBD': {'DT': 1, 'RB': 1},
            'NN': {'IN': 1},
            'IN': {'JJ': 1, 'DT': 1, 'CD': 1},
            'JJ': {'NNS': 1, 'WDT': 1},
            'NNS': {'VBN': 1},
            'VBN': {'IN': 1},
            'NNPS': {'IN': 1},
            'CD': {'CC': 1, ',': 1},
            'CC': {'CD': 1, 'DT': 1},
            ',': {'CC': 1},
            'WDT': {'VBD': 1},
            'RB': {'.': 1}
            }
        c_ti2_ti1_ti = {
            'DT': {
                'NNP': {'NNP': 1, 'NNPS': 1},
                'NN': {'IN': 1},
                'JJ': {'WDT': 1}},
            'NNP': {
                'NNP': {'VBD': 1},
                'VBD': {'DT': 1},
                'NNPS': {'IN': 1}},
            'VBD': {
                'DT': {'NN': 1},
                'RB': {'.': 1}},
            'NN': {
                'IN': {'JJ': 1}},
            'IN': {
                'JJ': {'NNS': 1},
                'DT': {'NNP': 1},
                'CD': {'CC': 1}},
            'JJ': {
                'NNS': {'VBN': 1},
                'WDT': {'VBD': 1}},
            'NNS': {
                'VBN': {'IN': 1}},
            'VBN': {
                'IN': {'DT': 1}},
            'NNPS': {
                'IN': {'CD': 1}},
            'CD': {
                'CC': {'CD': 1},
                ',': {'CC': 1}},
            'CC': {
                'CD': {',': 1},
                'DT': {'JJ': 1}},
            ',': {
                'CC': {'DT': 1}},
            'WDT': {
                'VBD': {'RB': 1}}
        }
        # true
        return (counts[0] == num_tokens, counts[1] == c_ti_wi, counts[2] == c_ti, counts[3] == c_ti1_ti, counts[4] == c_ti2_ti1_ti)
    return test

def test_compute_initial_distribution(test, order):
    """
    Input:
        - test: a string representing which test we are doing
    """
    # first sentence
    if test == 'a':
        if order == 2:
            init_dist = compute_initial_distribution(testa_list, 2)
            # true
            return init_dist == {'DT': 1.0}
        elif order == 3:
            init_dist = compute_initial_distribution(testa_list, 3)
            # true
            return init_dist == {'DT': {'NNP': 1.0}}
    return 0

def test_compute_emission_probabilities(test):
    """
    Input:
        - test: a string representing which test we are doing
    """
    # first sentence
    if test == 'a':
        counts = test_a_compute_counts
        emi_probs = compute_emission_probabilities(unique_words_a, unique_tags_a, counts[1], counts[2])
        emi_probs_desired = {
                    'DT': {'The': 0.25, 'a': 0.5, 'the': 0.25}, 
                    'NNP': {'New': 0.3333333333333333, 'Deal': 0.3333333333333333, 'United': 0.3333333333333333}, 
                    'VBD': {'was': 0.5, 'came': 0.5}, 
                    'NN': {'series': 1.0}, 
                    'IN': {'of': 0.3333333333333333, 'in': 0.3333333333333333, 'between': 0.3333333333333333}, 
                    'JJ': {'domestic': 0.5, 'few': 0.5}, 
                    'NNS': {'programs': 1.0}, 
                    'VBN': {'enacted': 1.0}, 
                    'NNPS': {'States': 1.0}, 
                    'CD': {'1933': 0.5, '1936': 0.5}, 
                    'CC': {'and': 1.0}, 
                    ',': {',': 1.0}, 
                    'WDT': {'that': 1.0}, 
                    'RB': {'later': 1.0}, 
                    '.': {'.': 1.0}
                }
        # true
        return emi_probs == emi_probs_desired
    return 0


# print(test_compute_counts('a'))
# print(test_compute_initial_distribution('a', 2))
#print(test_compute_emission_probabilities('a'))



#####################  TESTS  #####################
            # COMPUTE COUNTS:

# # test 0
# # WORKS
# test0_int = 2
# test_0 = compute_counts(test0_list, test0_int)
# #print("TEST 0: " + str(compute_counts(test0_list, test0_int)))

# # test 1
# # WORKS
# test1_int = 3
# # print("TEST 1: " + str(compute_counts(test0_list, test1_int)))

# # test a
# # first sentence
# #print("TEST A: ")
# test_a_compute_counts = compute_counts(testa_list, 3)
# unique_tags_a = list(compute_counts(testa_list, 3)[2].keys())
# # print("my unique tags: " + str(unique_tags_a))
# # print("𝐶(𝑡𝑖−2,𝑡𝑖−1,𝑡𝑖): " + str(test_a_compute_counts[4]))
# # print("𝐶(𝑡𝑖−1,𝑡𝑖)" + str(test_a_compute_counts[3]))

# # test b
# # first two sentences
# #print("TEST B: ")
# test_b_compute_counts = compute_counts(testb_list, 2)
# #compute_counts(testb_list, 3)

# # TEST ALL
# # runs
# test_all_compute_counts = compute_counts(test_all_list, 3)
# #print(test_all_compute_counts)

# ################
# #                COMPUTE INITIAL DISTRIBUTION:
# # TESTS

# # test 0
# # PASS
# #print(compute_initial_distribution(test0_list, test0_int))

# # test 1
# # PASS
# #print(compute_initial_distribution(test0_list, test1_int))

# # test a
# # first sentence
# # PASS
# # print(compute_initial_distribution(testa_list, 2))
# # # PASS
# # print(compute_initial_distribution(testa_list, 3))

# # test b
# # first and second sentence
# # PASS
# #print(compute_initial_distribution(testb_list, 2))
# #PASS
# #print(compute_initial_distribution(testb_list, 3))

# # TEST ALL
# # pass
# #print(compute_initial_distribution(test_all_list, 2))
# # pass
# #print(compute_initial_distribution(test_all_list, 3))

# ################
#             # COMPUTE EMISSION PROBABLIITIES:

# # tests
# # test 4
# unique_words4 = ['hw7', 'is', 'difficult', '.']
# unique_tags4 = ['N', 'V', 'A', '.']
# w_4 = test_0[1]
# c_4 = test_0[2]
# #print(compute_emission_probabilities(unique_words4, unique_tags4, w_4, c_4))

# # test a
# #print("poo")
# #print(compute_emission_probabilities(unique_words_a, unique_tags_a, test_a_compute_counts[1], test_a_compute_counts[2]))
# # print("Unique Tags: " + str(unique_tags_a))
# # # print(len(unique_tags_a))
# # print("")
# # print("Unique Words: " + str(unique_words_a))
# # # print(len(unique_words_a))
# # print("")
# # print("𝐶(𝑡𝑖,𝑤𝑖): " + str(test_a_compute_counts[1]))
# # print("")
# # print("𝐶(𝑡𝑖): " + str(test_a_compute_counts[2]))
# #test b
# #print(compute_emission_probabilities(unique_words_b, unique_tags_b, test_b_compute_counts[1], test_b_compute_counts[2]))

# # TEST ALL
# # pass
# #print(compute_emission_probabilities(unique_words_all, unique_tags_all, test_all_compute_counts[1], test_all_compute_counts[2]))

# ##############
# #                COMPUTE LAMBDAS:

# # pass
# #print(compute_lambdas(unique_tags_a, test_a_compute_counts[0], test_a_compute_counts[2], test_a_compute_counts[3], test_a_compute_counts[4], 3))
# # pass
# #print(compute_lambdas(unique_tags_a, test_a_compute_counts[0], test_a_compute_counts[2], test_a_compute_counts[3], test_a_compute_counts[4], 2))

# # TEST ALL
# # passes
# #print(compute_lambdas(unique_tags_all, test_all_compute_counts[0], test_all_compute_counts[2], test_all_compute_counts[3], test_all_compute_counts[4], 2))
# # passes
# #print(compute_lambdas(unique_tags_all, test_all_compute_counts[0], test_all_compute_counts[2], test_all_compute_counts[3], test_all_compute_counts[4], 3))

# ##############

# #                TRANSITION PROBABILIITY:
# #trans_matrix_a = transition_probs(testa_list, unique_tags_a, 26, 2, True)
# #print("trans matrix: " + str(trans_matrix_a))
# # passes
# #print(transition_probs(test_all_list, unique_tags_all, test_all_compute_counts[0], 2, True))
# # passes
# #print(transition_probs(test_all_list, unique_tags_all, test_all_compute_counts[0], 3, True))
# # passes
# #print(transition_probs(test_all_list, unique_tags_all, test_all_compute_counts[0], 2, False))
# # passes
# #print(transition_probs(test_all_list, unique_tags_all, test_all_compute_counts[0], 3, False))

# ##############

# #                BUILD HMM:

# # passes
# #hmma1 = build_hmm(testa_list, unique_tags_a, unique_words_a, 2, True)
# #print(hmma1)
# # passes
# #print(build_hmm_test(testa_list, unique_tags_a, unique_words_a, 3, True))

# # passes
# #print(build_hmm_test(testa_list, unique_tags_a, unique_words_a, 2, False))
# # passes
# #print(build_hmm_test(testa_list, unique_tags_a, unique_words_a, 3, False))

# # passes
# #print(build_hmm(test_all_list, unique_tags_all, unique_words_all, 2, True))
# # passes
# #print(build_hmm(test_all_list, unique_tags_all, unique_words_all, 3, True))
# # passes
# #print(build_hmm(test_all_list, unique_tags_all, unique_words_all, 2, False))
# # passes
# #print(build_hmm(test_all_list, unique_tags_all, unique_words_all, 3, False))

# ##############

