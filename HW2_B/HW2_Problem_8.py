'''
    DS 5230
    Summer 2022
    HW2_Problem_8_Hierarchical_Clustering

    Train and test hierarchical clustering on MNIST dataset
    Report performance by Purity and Gini Index

    Hongyan Yang
'''


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering as ac
from statistics import mode

CWD = os.getcwd()

def parse_MNIST(PATH):
    '''
    Parse the MNIST dataset
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(PATH)
    return x_train, y_train, x_test, y_test

def edit_normalize_MNIST(vectors):
    '''
    edit normalize the MNIST dataset
    '''
    out_vectors = vectors.reshape(vectors.shape[0], -1)
    out_vectors = out_vectors.astype("int16")
    out_vectors[out_vectors < 50] = -1
    out_vectors[out_vectors != -1] = 1
    return out_vectors

def plot_dendrograms(vectors, c_threshold):
    '''
    Create the dendrograms for the dataset
    '''
    plt.style.use("_mpl-gallery")
    plt.figure(figsize=(10, 7))
    plt.title("MNIST Dendograms")
    dend = shc.dendrogram(shc.linkage(vectors, method = "ward"),
                          color_threshold = c_threshold)
    plt.xticks([])
    plt.tight_layout()
    plt.show()

def get_true_labels(predicted_labels, labels, k):
    '''
    Get the "true" labels of predicted_labels
    '''
    labels_dict = dict()
    for i in range(k):
        labels_dict[i] = np.where(predicted_labels == i)[0]
    updated_labels = np.copy(predicted_labels)
    for i in range(k):
        updated_labels[labels_dict[i]] = mode(labels[labels_dict[i]])
    return updated_labels

def get_gini_index(predicted_labels, labels, k):
    '''
    Calculate the Gini Index to measure fitting performance
    '''
    labels_dict = dict()
    for i in range(k):
        labels_dict[i] = np.where(predicted_labels == i)[0]
    cluster_weight = np.stack([len(labels_dict[i]) / len(labels)
                               for i in range(k)], axis = 0)
    gini = np.stack([(1 - sum((np.unique(labels[labels_dict[i]],
                                         return_counts = True)[1]
                               / len(labels_dict[i])) ** 2))
                     for i in range(k)], axis = 0)
    return round(np.dot(cluster_weight, gini), 4)

def main():
    print("### A) Load and normalize the MNIST Dataset\n")
    train, train_labels, test, test_labels = parse_MNIST(CWD + "/mnist.npz")
    train, train_labels = train[:20000], train_labels[:20000]
    train, test = edit_normalize_MNIST(train), edit_normalize_MNIST(test)
    print(f"Shape of the training data is {train.shape}")
    print(f"Shape of the testing data is {test.shape}")
    print()
    print("### B) Create the dendrograms for the MNIST Dataset\n")
    plot_dendrograms(train, 350)
    plot_dendrograms(test, 233)
    print("### C) Fit the MNIST Dataset with hierarchical clustering\n")
    h_cluster = ac(n_clusters = 10, affinity = "euclidean", linkage = "ward")
    h_cluster.fit_predict(train)
    predicted_labels = get_true_labels(h_cluster.labels_, train_labels, 10)
    purity = round(sum(predicted_labels == train_labels) / len(train_labels), 4)
    gini_index = get_gini_index(h_cluster.labels_, train_labels, 10)
    print("## The training data's performance:\n")
    print(f"Purity: {purity}")
    print(f"Gini Index: {gini_index}")
    print()
    h_cluster.fit_predict(test)
    predicted_labels = get_true_labels(h_cluster.labels_, test_labels, 10)
    purity = round(sum(predicted_labels == test_labels) / len(test_labels), 4)
    gini_index = get_gini_index(h_cluster.labels_, test_labels, 10)
    print("## The testing data's performance:\n")
    print(f"Purity: {purity}")
    print(f"Gini Index: {gini_index}")


if __name__ == "__main__":
    main()
