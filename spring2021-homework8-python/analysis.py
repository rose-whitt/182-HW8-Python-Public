# Firstname Lastname
# NetID
# COMP 182 Spring 2021 - Homework 8, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...
#################   IMPORTS   #################
import math
import random
import numpy as np
#from matplotlib import *
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

def cut_sentences(data: list):
    """
    Input:
        training_data: a list of (word, POS-tag) pairs returned by the function read_pos_file
    
    Output:
        sentences: a list of lists where each element is a sentence and each element of the sentence
                    is the (word, tag)
    """

    sentences = []
    temp_sent = []

    for element in data:
        # print(element)
        if element[1] != '.':
            # print(element)
            temp_sent.append(element[0])
        if element[1] == '.':
            temp_sent.append(element[0])
            sentences.append(temp_sent)
            temp_sent = []
            
    return sentences

def first_words(sentences: list):
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

def unique_words_helper(x: list):
    words = []
    for i in x:
        if i[0] not in words:
            words.append(i[0])
    return words

def unique_tags_helper(x: list):
    tags = []
    for i in x:
        if i[1] not in tags:
            tags.append(i[1])
    return tags

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

def get_sentences(x: list):
    """
    Input:
        - x: a list where each element is the tuple (word, pos-tag)
    Output:
        a list of words without pos-tag

    """
    sentence = []
    for pair in x:
        sentence.append(pair[0])
    return sentence

#################### AUTOGRADER FUNCTIONS ####################

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
    num_tokens = len(training_data)
    c_ti_wi = defaultdict(lambda: defaultdict(int))
    c_ti = defaultdict(int)
    c_ti1_ti = defaultdict(lambda: defaultdict(int))
    if order == 3:
        c_ti2_ti1_ti = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
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

        #should i be checking that the tag3, tag2, tag1 combo exists?
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
    EPSILON = 0.00001
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for word in sentence:
        for tag in unique_tags:
            hmm.emission_matrix[tag][word] += EPSILON
    for tag in unique_tags:
        prob_sum = sum(hmm.emission_matrix[tag].values())
        for word in hmm.emission_matrix[tag].keys():
            hmm.emission_matrix[tag][word] /= prob_sum
 
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
    # print("INITIALIZATION")
    viterbi = defaultdict(lambda: defaultdict(int))
    backpointer = defaultdict(lambda: defaultdict(int))
    unique_tags = set(hmm.initial_distribution.keys()).union(set(hmm.transition_matrix.keys()))
    for tag in unique_tags:
        if (hmm.initial_distribution[tag] != 0) and (hmm.emission_matrix[tag][sentence[0]] != 0):
            viterbi[tag][0] = math.log(hmm.initial_distribution[tag]) + math.log(hmm.emission_matrix[tag][sentence[0]])
        else:
            viterbi[tag][0] = -1 * float('inf')

    # Dynamic programming.
    # print("DYNAMIC PROGRAMMING")
    for t in range(1, len(sentence)):
        backpointer["No_Path"][t] = "No_Path"
        for s in unique_tags:
            max_value = -1 * float('inf')
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

    # Termination
    # print("TERMINATION")
    max_value = -1 * float('inf')
    last_state = None
    final_time = len(sentence) - 1
    for s_prime in unique_tags:
        if viterbi[s_prime][final_time] > max_value:
            max_value = viterbi[s_prime][final_time]
            last_state = s_prime
    if last_state == None:
        last_state = "No_Path"

    # Traceback
    # print("TRACEBACK")
    tagged_sentence = []
    tagged_sentence.append((sentence[len(sentence)-1], last_state))
    for i in range(len(sentence)-2, -1, -1):
        next_tag = tagged_sentence[-1][1]
        curr_tag = backpointer[next_tag][i+1]   
        tagged_sentence.append((sentence[i], curr_tag))
    tagged_sentence.reverse()
    return tagged_sentence

def accuracy(expected: list, actual: list):
    """
    Inputs:
        - expected: a list of (word, POS-tag) pairs returned by the function read_pos_file reading testdat_tagged.txt
        - actual: a list of (word, POS-tag) pairs generated by viterbi
    
    Outputs:
        - accuracy_val: a float value
    """
    correct = 0
    num_tokens = len(actual)
    for idx, pair in enumerate(actual):
        if pair == expected[idx]:
            correct += 1
    accuracy_val = (correct / num_tokens)*100

    return accuracy_val

######################### GLOBAL VARIABLES: SET UP ##############################

# READ FILES
# print("READ FILES")
read_training = read_pos_file('training.txt')
read_testdata_tagged = read_pos_file('testdata_tagged.txt')

# GET LISTS FROM FILES
# print("GET LISTS FROM FILES")
training_word_pos_list = read_training[0]
len_training = len(training_word_pos_list)

test_word_pos_list = read_testdata_tagged[0]
len_test = len(test_word_pos_list)

# GET UNIQUE WORDS FROM FILES
# print("GET UNIQUE WORDS FROM FILES")
training_unique_words = read_training[1]
test_unique_words = read_testdata_tagged[1]

# GET UNIQUE TAGS FROM FILES
# print("GET UNIQUE TAGS FROM FILES")
training_unique_tags = read_training[2]
test_unique_tags = read_testdata_tagged[2]

# CUT SENTENCES
# print("CUT SENTENCES")
test_cut_sentences = cut_sentences(test_word_pos_list)
test_sentences = get_sentences(test_word_pos_list)
#print(test_cut_sentences)
######################### EXPERIMENT ONE ##############################

# CALCULATE PERCENTS
PERCENT_1 = math.floor(len_training / 100)
# PERCENT_5 = math.floor((5 * LEN_TRAINING) / 100)
# PERCENT_10 = math.floor((10 * LEN_TRAINING) / 100)
# PERCENT_25 = math.floor((25 * LEN_TRAINING) / 100)
# PERCENT_50 = math.floor((50 * LEN_TRAINING) / 100)
# PERCENT_75 = math.floor((75 * LEN_TRAINING) / 100)
# PERCENT_100 = math.floor((100 * LEN_TRAINING) / 100)

# TRAINING: 1%
# print("TRAINING 1%")
# training_1 = training_word_pos_list[0:PERCENT_1+1]
# uw_training_1 = unique_words_helper(training_1)
# ut_training_1 = unique_tags_helper(training_1)
# # INITIALIZE HMM
# print("HMM")
# training_hmm = build_hmm(training_1, ut_training_1, uw_training_1, 2, False)
# print("BUILD DONE")
# update_hmm(training_hmm, test_sentences)
# print("UPDATE DONE")
# vit_1 = bigram_viterbi(training_hmm, test_sentences)
# print("BIGRAM DONE")


def experiment_helper(percent: int, order: int, smoothing: bool):
    """
    Inputs:
        - percent: an int representing the percent of training desired
        - order: an int representing the order of hmm desired
        - smoothing: a boolean indicating whether or not to smooth the training corpus

    """
    # INITIALIZE TRAINING WITH PERCENT
    print("INITIALIZE TRAINING WITH PERCENT")
    if percent != 100:
        calc_percent = math.floor(percent * len_training / 100)
        training = training_word_pos_list[0:calc_percent+1]
        uw_training = unique_words_helper(training)
        ut_training = unique_tags_helper(training)
    else:
        training = training_word_pos_list
        uw_training = training_unique_words
        ut_training = training_unique_tags



    # CREATE HMM
    print("CREATE HMM")
    training_hmm = build_hmm(training, ut_training, uw_training, order, smoothing)
    print("BUILD HMM DONE")
    update_hmm(training_hmm, test_sentences)
    print("UPDATE HMM DONE")

    if order == 2:
        print("BEGIN VITERBI")
        vit = bigram_viterbi(training_hmm, test_sentences)
        print("VITERBI DONE")
        print("BEGIN ACCURACY")
        return accuracy(test_word_pos_list, vit)
    else:
        vit = trigram_viterbi(training_hmm, test_sentences)
        return accuracy(test_word_pos_list, vit)

def experiment(percents: list, order: int, smoothing: bool):
    """
    Inputs:
        - percents: a list of ints where each int is a desired percent to be calculate on training corpus
        - order: an int representing the order of hmm desired
        - smoothing: a boolean indicating whether or not to smooth the training corpus

    """
    accuracy_dict = {}
    for percent in percents:
        print("BEGIN " + str(percent) + "% CALCULATION.")
        accuracy = experiment_helper(percent, order, smoothing)
        print(str(percent) + "% ACCURACY: " + str(accuracy))
        print(str(percent) + "% CALCULATION DONE.")
        print("")
        accuracy_dict[percent] = accuracy
    return accuracy_dict

# print("BEGIN")
# print(experiment_helper(100, 2, False))
percents = [1, 5, 10, 25, 50, 75, 100]
# print("EXPERIMENT 1: ")
#experiment_1 = experiment(percents, 2, False)
#experiment_2 = experiment(percents, 3, False)
#experiment_3 = experiment(percents, 2, True)
#experiment_4 = experiment(percents, 3, True)

######################### ACCURACY RESULTS PLOT ##############################
import matplotlib.pyplot as plt
# print("here???????")
# # create data
x = [1, 5, 10, 25, 50, 75, 100]
y1 = [77.82340862422998, 88.09034907597535, 89.42505133470226, 92.50513347022587, 92.91581108829568, 93.73716632443532, 94.14784394250513]
y2 = [76.38603696098562, 87.47433264887063, 90.4517453798768, 93.32648870636551, 93.32648870636551, 93.73716632443532, 94.25051334702259]
y3 = [77.61806981519507, 88.39835728952772, 90.24640657084188, 93.01848049281314, 93.42915811088297, 94.45585215605749, 94.55852156057495]
y4 = [78.2340862422998, 88.80903490759754, 91.37577002053389, 93.73716632443532, 94.14784394250513, 94.55852156057495, 95.27720739219713]
# # plot line
# plt.scatter(x, y)
# plt.show()
# # plt.plot(x, y)
# # plt.show()
print()

# plt.scatter(x,y1)
plt.plot(x,y1, label="Bigram, No Smoothing")
# plt.scatter(x,y2)
plt.plot(x,y2, label="Trigram, No Smoothing")
plt.plot(x,y3, label="Bigram, Smoothing")
plt.plot(x,y4, label="Trigram, Smoothing")
plt.title("Experiments")
plt.xlabel("percent of training")
plt.ylabel("accuracy")
plt.legend()
# plt.legend([y1, y2, y3, y4], ["Bigram, No Smoothing", "Trigram, No Smoothing", "Bigram, Smoothing", "Trigram, Smoothing"])
plt.show()
# figure.tight_layout()