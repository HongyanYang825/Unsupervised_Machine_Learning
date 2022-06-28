'''
    DS 5230
    Summer 2022
    HW2_Problem_7_DBSCAN_on_real_data

    Train and test DBSCAN clustering on three real datasets

    Hongyan Yang
'''


import numpy as np
import pandas as pd

from HW2_Problem_5 import *
from HW2_Problem_6 import *
from HW2_Problem_8 import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_20_NG(NG_DATA, max_features = 5000, use_idf = False):
    '''
    Parse and normalize the 20NG dataset
    '''
    # Define the vectorizer
    vectorizer = TfidfVectorizer(decode_error = "replace", stop_words = "english",
                                 max_features = max_features, use_idf = use_idf)
    # Transform a list of raw texts to a matrix of tf values
    vectors = vectorizer.fit_transform(NG_DATA)                         
    return np.asarray(vectors.todense())

def normalize_zero_mean_unit_variance(vectors):
    '''
    Normalize dataset with zero column mean and unit column std
    '''
    col_mean, col_std = np.mean(vectors, axis = 0), np.std(vectors, axis = 0)
    out_vectors = (vectors - col_mean) / col_std
    return out_vectors

def main():
    print("### A) Load and parse the 20NG dataset\n")
    ng_data = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                 subset = "test")
    ng_dataset, ng_labels = ng_data.data, ng_data.target
    vectors = parse_20_NG(ng_dataset, max_features = 5000, use_idf = True)
    vectors = normalize_zero_mean_unit_variance(vectors)
    indices = np.random.choice(vectors.shape[0], 1000, replace = False)
    vectors, ng_labels = vectors[indices], ng_labels[indices]
    ng_df = pd.DataFrame()
    ng_df["pt"] = range(vectors.shape[0])
    print(ng_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(ng_df, vectors, 3, 5, 50)
    #optimize_parameters(ng_df, vectors, 10, 5, 50)
    #optimize_parameters(ng_df, vectors, 30, 5, 50)
    #optimize_parameters(ng_df, vectors, 100, 5, 50)
    #optimize_parameters(ng_df, vectors, 300, 5, 50)
    #optimize_parameters(ng_df, vectors, 1000, 5, 50)
    #optimize_parameters(ng_df, vectors, 3000, 5, 50)
    #optimize_parameters(ng_df, vectors, 10000, 5, 50)
    print("After analyzing the plots, Epsilon between 80 and 100 is appropriate")
    print()
    print("## Apply DBSCAN to cluster dataset\n")
    dbscan_dict = get_neighbors(vectors, 90)
    k, l_c, noise = get_dbscan_clusters(ng_df, dbscan_dict, 10)
    print(ng_df.head())
    print()
    print(f"DBSCAN classifies {k} clusters, the largest component: {l_c:.2f}, noise: {noise:.2f}")
    print()
    print("## The DBSCAN classification's performance:\n")
    predicted_labels = get_true_labels(ng_df["cluster"].to_numpy(), ng_labels, 1)
    purity = round(sum(predicted_labels == ng_labels) / len(ng_labels), 4)
    gini_index = get_gini_index(ng_df["cluster"].to_numpy(), ng_labels, 1)
    print(f"Purity: {purity}")
    print(f"Gini Index: {gini_index}")
    print("\n")
    print("### B) Load and parse the FASHION dataset\n")
    fashion_data = np.genfromtxt("fashion-mnist_test.csv", delimiter=",")
    fashion_data = np.delete(fashion_data, 0, 0)
    fashion_labels = fashion_data[:, 0]
    vectors = edit_normalize_MNIST(np.delete(fashion_data, 0, 1))
    indices = np.random.choice(vectors.shape[0], 3000, replace = False)
    vectors, fashion_labels = vectors[indices], fashion_labels[indices]
    fashion_df = pd.DataFrame()
    fashion_df["pt"] = range(vectors.shape[0])
    print(fashion_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(fashion_df, vectors, 3, 2, 20)
    #optimize_parameters(fashion_df, vectors, 10, 2, 20)
    #optimize_parameters(fashion_df, vectors, 30, 2, 20)
    #optimize_parameters(fashion_df, vectors, 100, 2, 20)
    #optimize_parameters(fashion_df, vectors, 300, 2, 20)
    #optimize_parameters(fashion_df, vectors, 1000, 2, 20)
    print("After analyzing the plots, Epsilon between 15 and 20 is appropriate")
    print()
    print("## Apply DBSCAN to cluster dataset\n")
    dbscan_dict = get_neighbors(vectors, 17.5)
    k, l_c, noise = get_dbscan_clusters(fashion_df, dbscan_dict, 3)
    print(fashion_df.head())
    print()
    print(f"DBSCAN classifies {k} clusters, the largest component: {l_c:.2f}, noise: {noise:.2f}")
    print()
    print("## The DBSCAN classification's performance:\n")
    num_clusters = int(input("Enter number of clusters: "))
    predicted_labels = get_true_labels(fashion_df["cluster"].to_numpy(), fashion_labels, num_clusters)
    purity = round(sum(predicted_labels == fashion_labels) / len(fashion_labels), 4)
    gini_index = get_gini_index(fashion_df["cluster"].to_numpy(), fashion_labels, 1)
    print(f"Purity: {purity}")
    print(f"Gini Index: {gini_index}")
    print("\n")
    print("### C) Load and parse the HouseHold dataset\n")
    hh_data = np.genfromtxt("household_power_consumption.txt", delimiter=";")
    hh_data = np.delete(hh_data, 0, axis = 0)
    hh_data = np.delete(hh_data, [0, 1], axis = 1)
    hh_data = hh_data[~np.isnan(hh_data).any(axis = 1)]
    vectors = normalize_zero_mean_unit_variance(hh_data)
    indices = np.random.choice(vectors.shape[0], 3000, replace = False)
    vectors = vectors[indices]
    hh_df = pd.DataFrame()
    hh_df["pt"] = range(vectors.shape[0])
    print(hh_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(hh_df, vectors, 3, 0.2, 20)
    #optimize_parameters(hh_df, vectors, 6, 0.2, 20)
    #optimize_parameters(hh_df, vectors, 12, 0.2, 20)
    #optimize_parameters(hh_df, vectors, 24, 0.2, 20)
    print("After analyzing the plots, Epsilon between 1 and 2 is appropriate")
    print()
    print("## Apply DBSCAN to cluster dataset\n")
    dbscan_dict = get_neighbors(vectors, 1.5)
    k, l_c, noise = get_dbscan_clusters(hh_df, dbscan_dict, 5)
    print(hh_df.head())
    print()
    print(f"DBSCAN classifies {k} clusters, the largest component: {l_c:.2f}, noise: {noise:.2f}")

 
if __name__ == "__main__":
    main()
