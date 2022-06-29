'''
    DS 5230
    Summer 2022
    HW3A_Problem_1_Supervised_Classification

    Train and test L2-reg Logistic Regression and Decision Trees
    on MNIST, Spambase and 20NG datasets
    Analyze the model trained in terms of features

    Hongyan Yang
'''


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics

CWD = os.getcwd()

def parse_MNIST(path):
    '''
    Parse the MNIST dataset
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path)
    return x_train, y_train, x_test, y_test

def parse_spambase_names(path):
    '''
    Parse the Spambase dataset's features' names
    '''
    names = list()
    with open(path, encoding = "utf-8") as f:
        for line in f:
            if (line.startswith("word") or line.startswith("char")
                or line.startswith("capital")):
                names.append(line.split(":")[0])
    return np.array(names)

def parse_20_NG_names(vectorizer):
    '''
    Parse the 20NG dataset's features' names
    '''
    names, names_dict = list(), dict()
    for key in vectorizer.vocabulary_:
        value = vectorizer.vocabulary_[key]
        names_dict[value] = key
    for i in range(len(names_dict)):
        names.append(names_dict[i])
    return np.array(names)

def parse_20_NG(ng_data, max_features = 5000, use_idf = False):
    '''
    Parse the 20NG dataset
    '''
    # Define the vectorizer
    vectorizer = TfidfVectorizer(decode_error = "replace", stop_words = "english",
                                 max_features = max_features, use_idf = use_idf)
    # Transform a list of raw texts to a matrix of tf values
    vectors = vectorizer.fit_transform(ng_data)
    names = parse_20_NG_names(vectorizer)
    return np.asarray(vectors.todense()), names

def normalize_zero_mean_unit_variance(vectors):
    '''
    Normalize dataset with zero column mean and unit column std
    '''
    col_mean, col_std = np.mean(vectors, axis = 0), np.std(vectors, axis = 0)
    out_vectors = (vectors - col_mean) / col_std
    return out_vectors

def edit_normalize_MNIST(vectors):
    '''
    edit normalize the MNIST dataset
    '''
    out_vectors = vectors.reshape(vectors.shape[0], -1)
    out_vectors[out_vectors < 50] = 0
    out_vectors[out_vectors != 0] = 1
    return out_vectors

def print_tree_nodes(clf, num_splits = 30, names = None, show_names = False):
    '''
    Analyze features and print the first 30 splits of the decision trees
    '''
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_depth = np.zeros(shape = n_nodes, dtype = np.int64)
    is_leaves = np.zeros(shape = n_nodes, dtype = bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # Node splits if the left and right child of the node is not the same
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    for i in range(num_splits):
        if is_leaves[i]:
            print("node {node} is a leaf node.".format(node = i))
        else:
            if show_names is True:
                print("node {node} is a split node: "
                      "go to node {left} if {feature} <= {threshold} "
                      "else to node {right}.".format(node = i,
                                                     left = children_left[i],
                                                     feature = names[feature[i]],
                                                     threshold = threshold[i],
                                                     right = children_right[i]))
            else:
                print("node {node} is a split node: "
                      "go to node {left} if X[:, {feature}] <= {threshold} "
                      "else to node {right}.".format(node = i,
                                                     left = children_left[i],
                                                     feature = feature[i],
                                                     threshold = threshold[i],
                                                     right = children_right[i]))

def logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0,
                     max_iter = 800, confusion_matrix = False, names = None,
                     show_names = False):
    '''
    Train and test L2-reg Logistic Regression on the dataset and analyze features
    '''
    clf = LogisticRegression(penalty = "l2", solver = "lbfgs", tol = tol,
                             max_iter = max_iter)
    clf.fit(train, train_labels)
    if confusion_matrix is True:
        predictions = clf.predict(test)
        cm = metrics.confusion_matrix(predictions, test_labels)
        plt.figure(figsize = (9,9))
        sns.heatmap(cm, annot = True, fmt = ".3f", linewidths = .5, square = True,
                    cmap = "Blues_r")
        plt.ylabel("Predicted label")
        plt.xlabel("Actual label")
        plt.title("Accuracy Score: {0}".format(score), size = 15)
        plt.show()
    score = clf.score(test, test_labels)
    print(f"Purity: {score}\n")
    coefficients_indices = np.argsort(-np.absolute(clf.coef_), axis = 1)[:, :30]
    print("The highest-absolute-value 30 coefficients are as follows: \n")
    if show_names is True:
        print([names[i] for i in coefficients_indices])
    else:
        print(coefficients_indices)
    return score

def decision_tree_clf(train, train_labels, test, test_labels, min_leaf = 1,
                      names = None, show_names = False):
    '''
    Train and test Decision Trees on the dataset and analyze important features
    '''
    clf = tree.DecisionTreeClassifier(min_samples_leaf = min_leaf, random_state = 0)
    clf = clf.fit(train, train_labels)
    score = clf.score(test, test_labels)
    print(f"Purity: {score}\n")
    print("The first 30 splits of the decision trees are as follows: \n")
    print_tree_nodes(clf, names = names, show_names = show_names)

def main():
    print("### A) Load and normalize the MNIST Dataset\n")
    train, train_labels, test, test_labels = parse_MNIST(CWD + "/mnist.npz")
    train, test = edit_normalize_MNIST(train), edit_normalize_MNIST(test)
    print("## Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 800)
    print()
    print("## Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels)
    print("\n")
    print("### B) Load and normalize the Spambase Dataset\n")
    spambase_data = np.loadtxt("spambase.data", delimiter=",", dtype = float)
    labels = spambase_data[:, -1]
    spambase_data = np.delete(spambase_data, -1, 1)
    spambase_data = normalize_zero_mean_unit_variance(spambase_data)
    data = train_test_split(spambase_data, labels, test_size = 0.25, random_state = 0)
    train, test, train_labels, test_labels = data
    names = parse_spambase_names(CWD + "/spambase.names")
    print("## Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 1000,
                     names = names, show_names = True)
    print()
    print("## Train and test Decision Trees and analyze features:\n")
    decision_tree_clf(train, train_labels, test, test_labels,
                      names = names, show_names = True)
    print("\n")
    print("### C) Load and normalize the 20NG Dataset\n")
    ng_set = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                subset = "all")
    ng_data, names = parse_20_NG(ng_set.data, max_features = 5000, use_idf = True)
    data = train_test_split(ng_data, ng_set.target, test_size = 0.25,
                            random_state = 0)
    train, test, train_labels, test_labels = data
    print("## Train and test L2-reg Logistic Regression and analyze features:\n")
    logistic_reg_clf(train, train_labels, test, test_labels, tol = 1e0, max_iter = 1000,
                     names = names, show_names = True)
    print()
    print("## Train and test Decision Trees and analyze features:\n")
    print("# Decision Trees performance for number of leaves = 1:\n")
    decision_tree_clf(train, train_labels, test, test_labels, min_leaf = 1,
                      names = names, show_names = True)
    print()
    print("# Decision Trees performance for number of leaves = 5:\n")
    decision_tree_clf(train, train_labels, test, test_labels, min_leaf = 5,
                      names = names, show_names = True)


if __name__ == "__main__":
    main()
