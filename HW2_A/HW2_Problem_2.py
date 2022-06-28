'''
    DS 5230
    Summer 2022
    HW2_Problem_2_K_Means_on_data

    Train and test K_Means classification for 20NG, MNIST and
    FASHION datasets
    Report performance by Purity and Gini Index

    Hongyan Yang
'''


import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from statistics import mode
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
   
CWD = os.getcwd()
PATH = CWD + "/mnist.npz"

def parse_20_NG(NG_DATA, max_features = 5000, use_idf = False):
    '''
    Return a parsed matrix of tf values
    '''
    # Define the vectorizer
    vectorizer = TfidfVectorizer(decode_error = "replace", stop_words = "english",
                                 max_features = max_features, use_idf = use_idf)
    # Transform a list of raw texts to a matrix of tf values
    vectors = vectorizer.fit_transform(NG_DATA)                         
    return vectors.todense()

def visualize_MNIST(vector):
    '''
    Plot a specific instance in greyscale
    '''
    plt.imshow(vector, cmap = plt.cm.binary)
    plt.show()

def normalize_shift_and_scale(vectors):
    '''
    Return normalized dataset with each row values between zero and one
    '''
    # Reshape the parsed MNIST dataset into a 2D matrix
    out_vectors = vectors.reshape(vectors.shape[0], -1)
    row_max = np.max(out_vectors, axis = 1)
    out_vectors = (np.divide(out_vectors.T, row_max)).T
    out_vectors = out_vectors.reshape(vectors.shape)
    return out_vectors

def get_memberships_dict(memberships, max_label):
    '''
    Get the indices of members of each current cluster
    '''
    memberships_dict = dict()
    for i in range(max_label + 1):
        memberships_dict[i] = np.where(memberships == i)[0]
    return memberships_dict

def get_true_memberships(memberships, max_label, labels):
    '''
    Get true labels of members under current clustering
    '''
    memberships_dict = get_memberships_dict(memberships, max_label)
    for i in range(len(memberships_dict)):
        memberships[memberships_dict[i]] = mode(labels[memberships_dict[i]])
    return memberships

def get_gini_index(memberships, max_label, labels):
    '''
    Calculate the Gini Index to measure fitting performance
    '''
    memberships_dict = get_memberships_dict(memberships, max_label)
    cluster_weight = np.stack([len(memberships_dict[i]) / len(labels)
                               for i in range(max_label + 1)], axis = 0)
    gini = np.stack([(1 - sum((np.unique(labels[memberships_dict[i]],
                                        return_counts = True)[1]
                              / len(memberships_dict[i])) ** 2))
                     for i in range(max_label + 1)], axis = 0)
    return round(np.dot(cluster_weight, gini), 4)
                              
def initial_assign(vectors, max_label):
    '''
    Make initial (random) cluster assignment of each observation 
    '''
    memberships = np.random.randint(max_label + 1, size = len(vectors))
    return memberships

def m_step(vectors, memberships, max_label):
    '''
    M step of the K_Means clustring algorithm
    '''    
    centroids = np.stack([np.average(vectors[np.where(memberships == i)], axis = 0)
                         for i in range(max_label + 1)], axis = 0)
    return centroids

def e_step(vectors, centroids, cosine_dist = True):
    '''
    E step of the K_Means clustring algorithm
    '''
    if cosine_dist:
        # Get the index of each instance's nearest centroid
        memberships = np.argmax(cosine_similarity(np.asarray(vectors),
                                                  np.asarray(centroids)), axis = 1)
    else:
        memberships = np.argmin(euclidean_distances(np.asarray(vectors),
                                                    np.asarray(centroids)), axis = 1)
    return memberships

def k_means_objective(vectors, centroids, memberships, max_label):
    '''
    Objective function of the K_Means clustring algorithm
    '''
    objectives = np.stack([sum(euclidean_distances(vectors[np.where(memberships == i)],
                                                   centroids[i].reshape(1, -1)) ** 2)
                           for i in range(max_label + 1)], axis = 0)
    objective = round(sum(objectives)[0], 4)
    return objective

def k_means_fit(vectors, labels, max_label, tol = 1e-4, max_iter = 300,
                cosine_distances = True):
    '''
    Fit K_Means clustring algorithm to datasets
    '''
    iter, purity_list, gini_list, objective_list = 0, list(), list(), list()
    while iter < max_iter:
        memberships, centroids_diff = initial_assign(vectors, max_label), 1
        centroids = m_step(vectors, memberships, max_label)
        while centroids_diff > tol:
            memberships = e_step(vectors, centroids, cosine_dist = cosine_distances)
            new_centroids = m_step(vectors, memberships, max_label)
            centroids_diff = abs(np.linalg.norm(new_centroids) - np.linalg.norm(centroids))
            centroids = new_centroids
        try:
            objective = k_means_objective(vectors, new_centroids, memberships, max_label)
            gini_index = get_gini_index(memberships, max_label, labels)
            memberships = get_true_memberships(memberships, max_label, labels)
            purity = round(sum(memberships == labels) / len(labels), 4)
            gini_list.append(gini_index)
            purity_list.append(purity)
            objective_list.append(objective)
            iter += 1
        except:
            continue
    min_gini_index, max_purity = min(gini_list), max(purity_list) 
    min_objective = min(objective_list)
    return min_gini_index, max_purity, min_objective

def main():
    print("### A) run K_Means on the MNIST Dataset\n")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(PATH)
    #data_m = x_test.reshape(x_test.shape[0], -1)
    data_m = normalize_shift_and_scale(x_test).reshape(x_test.shape[0], -1)
    labels_m = y_test
    print("## K_Means performance for K = 10")
    result_m = k_means_fit(data_m, labels_m, 9, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 5")
    result_m = k_means_fit(data_m, labels_m, 4, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 20")
    result_m = k_means_fit(data_m, labels_m, 19, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print("\n")
    print("### B) run K_Means on the FASHION Dataset\n")
    data_f = np.genfromtxt('fashion-mnist_test.csv', delimiter = ',')
    data_f = np.delete(data_f, 0, 0)
    labels_f = data_f[:, 0]
    data_f = np.delete(data_f, 0, 1)
    data_f = normalize_shift_and_scale(data_f).reshape(data_f.shape[0], -1)
    print("## K_Means performance for K = 10")
    result_m = k_means_fit(data_f, labels_f, 9, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 5")
    result_m = k_means_fit(data_f, labels_f, 4, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 20")
    result_m = k_means_fit(data_f, labels_f, 19, tol = 1e-4, max_iter = 300,
                           cosine_distances = False)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print("\n")
    print("### C) run K_Means on the 20NG Dataset\n")
    ng_dataset = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                    subset = "test")
    data_ng, labels_ng = ng_dataset.data, ng_dataset.target
    data_ng = parse_20_NG(data_ng, max_features = 5000, use_idf = True)
    print("## K_Means performance for K = 20")
    result_m = k_means_fit(data_ng, labels_ng, 19, tol = 1e-4, max_iter = 300,
                           cosine_distances = True)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 10")
    result_m = k_means_fit(data_ng, labels_ng, 9, tol = 1e-4, max_iter = 300,
                           cosine_distances = True)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")
    print()
    print("## K_Means performance for K = 40")
    result_m = k_means_fit(data_ng, labels_ng, 39, tol = 1e-4, max_iter = 300,
                           cosine_distances = True)
    print(f"Objective: {result_m[2]}")
    print(f"Purity: {result_m[1]}")
    print(f"Gini Index: {result_m[0]}")


if __name__ == "__main__":
    main()
