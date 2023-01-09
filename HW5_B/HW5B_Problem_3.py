'''
    DS 5230
    Summer 2022
    HW5B_Problem_3_Implement_LDA

    Implement Latent Dirichlet Allocation using Gibbs Sampling

    Hongyan Yang
'''


import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from wordcloud import WordCloud

porter = PorterStemmer()
lancaster = LancasterStemmer()

CWD = os.getcwd()
PATH = CWD + "/sonnets.txt"

def stem_sentence(sentence):
    '''
    Stem every word in a sentence. Return the updated sentence
    '''
    token_words = word_tokenize(sentence)
    token_words = list(filter(lambda x: len(x) > 2, token_words))
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(lancaster.stem(word))
    return " ".join(stem_sentence)

def parse_raw_data(file_path):
    '''
    Read the raw data file and create a D * W matrix X,
    X[d,w] = count of word w in doc d
    '''
    with open(file_path, "r", encoding="utf-8") as f:
        D_list = f.readlines()
        D_list = list(map(lambda x: x.strip("\n"), D_list))
    # Extract tf features for the Document matrix
    vct = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
    D = vct.fit_transform(D_list).todense()
    Vocab = vct.get_feature_names_out()
    return D, Vocab

def d_to_words(D, Vocab, i):
    '''
    Convert a row in D * W matrix to a list of string words
    '''
    w_indices = np.nonzero(D[i])[1]
    w_counts = np.asarray(D[i]).reshape(-1)[w_indices]
    w_list = Vocab[w_indices]
    words_list = list()
    for i in range(len(w_indices)):
        words_list.extend(w_counts[i] * [w_list[i]])
    return words_list

def get_DOCS_and_Z(D, Vocab):
    '''
    Get DOCS matrix and initial Z matrix
    '''
    DOCS, Z = list(), list()
    for i in range(len(D)):
        words_list = d_to_words(D, Vocab, i)
        DOCS.append(words_list)
        Z.append([-1] * len(words_list))
    return DOCS, Z

def LDA(D, Vocab, K = 6, T = 1000):
    '''
    Implement Latent Dirichlet Allocation using Gibbs Sampling
    '''
    DOCS, Z = get_DOCS_and_Z(D, Vocab)
    A = 5 * np.ones((len(D), K))
    B = 0.01 * np.ones((K, len(Vocab)))
    BSUM = np.sum(B, axis=1)
    for t in range(T):
        for d in range(len(D)):
           for i in range(len(DOCS[d])):
               w = DOCS[d][i]
               z = Z[d][i]
               B_index = Vocab.tolist().index(w)
               # Subtract current topic z from counts
               if z != -1:
                   A[d, z] -= 1
                   B[z, B_index] -= 1
                   BSUM[z] -= 1
               # Prepare Gibbs-sampling conditional distribution over topics
               dst = A[d, :] * B[:, B_index] / BSUM
               p = dst / np.sum(dst)
               new_z = np.random.choice(range(K), 1, p=p)[0]
               # Update Z and counts
               Z[d][i] = new_z
               A[d, z] += 1
               B[new_z, B_index] += 1
               BSUM[new_z] += 1
    return B

def plot_wordcloud(Vocab, B):
    '''
    Display a "wordcloud" for each topic, using B for word weights
    '''
    plot = WordCloud(background_color='white', width=300, height=300, margin=2)
    for k in range(len(B)):
        words_freq = {k: v for k, v in zip(Vocab, B[k])}
        plot.fit_words(words_freq)
        plot.to_file(f"wordcloud_topic_{k}.png")

def main():
    T = int(input("Please enter the number of iterations: "))
    D, Vocab = parse_raw_data(PATH)
    B = LDA(D, Vocab, K = 6, T = T)
    plot_wordcloud(Vocab, B)


#if __name__ == "__main__":
    #main()
