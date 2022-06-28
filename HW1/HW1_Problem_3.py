'''
    DS 5230
    Summer 2022
    HW1_Problem_3_MNIST_20_NG_Parse_Normalize_Pairwaise_Similarity

    Parse and normalize the 20NG and MNIST datasets
    Calculate vectors distances to get pairwise similarity matrix

    Hongyan Yang
'''


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

CWD = os.getcwd()
PATH = CWD + "/mnist.npz"

def parse_20_NG(NG_DATA, max_features = 5000, use_idf = False):
    '''
    Function -- parse_20_NG
    Parse and normalize the 20NG dataset
    Parameters: NG_DATA (list) -- a list of the raw texts of newsgroups posts
                max_features (int) -- max number of features
                use_idf (bool) -- number of neighbors in the KNN model
                cosine_distances (bool) -- enable inverse-document-frequency or not
    Return a parsed matrix of tf values
    '''
    # Define the vectorizer
    vectorizer = TfidfVectorizer(decode_error = "replace", stop_words = "english",
                                 max_features = max_features, use_idf = use_idf)
    # Transform a list of raw texts to a matrix of tf values
    vectors = vectorizer.fit_transform(NG_DATA)                         
    return vectors.todense()

def parse_MNIST(PATH):
    '''
    Function -- parse_MNIST
    Parse the MNIST dataset
    Parameters: PATH (str) -- local PATH of the MNIST dataset
    Return a bunch of parsed training and testing datasets
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(PATH)
    return x_train, y_train, x_test, y_test

def visualize_MNIST(vector):
    '''
    Function -- visualize_MNIST
    Visualize a specific instance of the MNIST dataset
    Parameters: vector (matrix) -- an instance matrix of handwriting number
    Plot a specific instance in greyscale
    '''
    plt.imshow(vector, cmap = plt.cm.binary)
    plt.show()

def normalize_zero_mean_unit_variance(vectors):
    '''
    Function -- normalize_zero_mean_unit_variance
    Normalize the 20NG dataset
    Parameters: vectors (matrix) -- parsed 20NG dataset
    Return normalized dataset with zero column mean and unit column std
    '''
    col_mean, col_std = np.mean(vectors, axis = 0), np.std(vectors, axis = 0)
    out_vectors = (vectors - col_mean) / col_std
    return out_vectors

def normalize_shift_and_scale(vectors):
    '''
    Function -- normalize_shift_and_scale
    Normalize the MNIST dataset
    Parameters: vectors (matrix) -- parsed MNIST dataset
    Return normalized dataset with each row values between zero and one
    '''
    # Reshape the parsed MNIST dataset into a 2D matrix
    out_vectors = vectors.reshape(vectors.shape[0], -1)
    row_max = np.max(out_vectors, axis = 1)
    out_vectors = (np.divide(out_vectors.T, row_max)).T
    out_vectors = out_vectors.reshape(vectors.shape)
    return out_vectors

def edit_distances_20_NG(vectors):
    '''
    Function -- edit_distances_20_NG
    Calculate edit distances of the 20NG dataset
    Parameters: vectors (matrix) -- parsed 20NG dataset
    Return a pairwise similarity matrix of edit distance values
    '''
    vectors[vectors != 0] = 1
    vectors_distances = np.dot(vectors, vectors.T)
    return vectors_distances

def edit_distances_MNIST(vectors):
    '''
    Function -- edit_distances_MNIST
    Calculate edit distances of the MNIST dataset
    Parameters: vectors (matrix) -- parsed MNIST dataset
    Return a pairwise similarity matrix of edit distance values
    '''
    out_vectors = vectors.reshape(vectors.shape[0], -1)
    # Change matrix datatype to allow negative values
    out_vectors = out_vectors.astype("int16")
    # Set the edit distance threshold to 50
    out_vectors[out_vectors < 50] = -1
    out_vectors[out_vectors != -1] = 1
    vectors_distances = (784 - np.dot(out_vectors, out_vectors.T)) / 2
    return vectors_distances

def get_cosine_distances(vectors):
    '''
    Function -- get_cosine_distances
    Calculate cosine distances of the input dataset
    Parameters: vectors (matrix) -- input dataset
    Return a pairwise similarity matrix of cosine distance values
    '''
    return cosine_similarity(np.asarray(vectors))

def get_euclidian_distances(vectors):
    '''
    Function -- get_euclidian_distances
    Calculate euclidian distances of the input dataset
    Parameters: vectors (matrix) -- input dataset
    Return a pairwise similarity matrix of euclidian distance values
    '''
    return euclidean_distances(np.asarray(vectors))

def get_manhattan_distances(vectors):
    '''
    Function -- get_manhattan_distances
    Calculate manhattan distances of the input dataset
    Parameters: vectors (matrix) -- input dataset
    Return a pairwise similarity matrix of manhattan distance values
    '''
    return manhattan_distances(np.asarray(vectors))

def main():
    NG_DATA = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                 subset = "test").data
    mnist_vectors = parse_MNIST(PATH)[2]
    ng_vectors = parse_20_NG(NG_DATA, max_features = 5000, use_idf = False)
    print("Zero mean, unit variance normalization:\n")
    print("Original 20NG dataset matrix:")
    print(ng_vectors)
    print("\nNormalized 20NG dataset matrix:")
    out_vectors = normalize_zero_mean_unit_variance(ng_vectors)
    print(normalize_zero_mean_unit_variance(out_vectors))
    print("\nColumn means after normalization:")
    print(np.mean(out_vectors, axis = 0))
    print("\nColumn stds after normalization:")
    print(np.std(out_vectors, axis = 0))
    print("\n\nShift and scale normalization:\n")
    print("Original MNIST dataset matrix:")
    out_vectors = mnist_vectors.reshape(mnist_vectors.shape[0], -1)
    print(out_vectors)
    print("\nNormalized MNIST dataset matrix:")
    out_vectors = normalize_shift_and_scale(out_vectors)
    print(out_vectors)
    print("\nRow maximums after normalization:")
    print(np.max(out_vectors, axis = 1)[:20])
    print("\nRow minimums after normalization:")
    print(np.min(out_vectors, axis = 1)[:20])


if __name__ == "__main__":
    main()
