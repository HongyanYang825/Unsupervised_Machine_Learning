'''
    DS 5230
    Summer 2022
    HW3B_Problem_3_HARR_features_for_MNIST

    Train and test feature selection using HAAR feature extraction
    on the MNIST dataset
    
    Hongyan Yang
'''


import os
import warnings
import numpy as np
import tensorflow as tf

from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

CWD = os.getcwd()

def parse_MNIST(path):
    '''
    Parse the MNIST dataset
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path)
    return x_train, y_train, x_test, y_test

def edit_normalize_MNIST(vectors):
    '''
    Edit normalize the MNIST dataset
    '''
    out_vectors = np.copy(vectors)
    out_vectors[out_vectors < 50] = 0
    out_vectors[out_vectors != 0] = 1
    return out_vectors

def get_sum_matrix(vectors, size):
    '''
    Construct summation matrices for images in the MNIST
    '''
    sum_mat = np.zeros(vectors.shape)
    sum_mat[:, 0, 0] = vectors[:, 0, 0]
    for i in range(1, size):
         sum_mat[:, i, 0] = sum_mat[:, i - 1, 0] + vectors[:, i, 0]
    for j in range(1, size):
         sum_mat[:, 0, j] = sum_mat[:, 0, j - 1] + vectors[:, 0, j]
    for i in range(1, size):
        for j in range(1, size):
            sum_mat[:, i, j] = (sum_mat[:, i - 1, j] + sum_mat[:, i, j - 1]
                                - sum_mat[:, i - 1, j - 1] + vectors[:, i, j])
    return sum_mat

def get_random_cut(size, cut_num, min_area = 150):
    '''
    Randomly generate cut_num rectangles fitting inside the image box
    '''
    cut_mat = np.zeros(shape = (cut_num, 6, 2))
    for i in range(cut_num):
        adj_coord = np.zeros(shape = (2, 2))
        coord = np.random.randint(size, size = (2, 2))
        coord_diff = np.absolute(coord[0, :] - coord[1, :])
        while (np.multiply(coord_diff[0], coord_diff[1]) < min_area
               or np.sum(coord_diff % 2) != 0):
            coord = np.random.randint(size, size = (2, 2))
            coord_diff = np.absolute(coord[0, :] - coord[1, :])
        adj_coord[0, :] = np.min(coord, axis = 0)
        adj_coord[1, :] = np.max(coord, axis = 0)
        cut_mat[i][:2, :] = adj_coord
        cut_mat[i][2, 0] = (adj_coord[0, 0] + adj_coord[1, 0]) / 2
        cut_mat[i][2, 1] = adj_coord[1, 1]
        cut_mat[i][3, 0] = cut_mat[i][2, 0]
        cut_mat[i][3, 1] = adj_coord[0, 1]
        cut_mat[i][4, 0] = adj_coord[1, 0]
        cut_mat[i][4, 1] = (adj_coord[0, 1] + adj_coord[1, 1]) / 2
        cut_mat[i][5, 0] = adj_coord[0, 0]
        cut_mat[i][5, 1] = cut_mat[i][4, 1]
    cut_mat = cut_mat.astype(int)
    return cut_mat

def get_cut_area(sum_mat, coord):
    '''
    Compute the area of a generated rectangle inside the image box
    '''
    area_3 = sum_mat[:, coord[1, 0], coord[1, 1]]
    area_0 = sum_mat[:, coord[0, 0], coord[0, 1]]
    area_1 = sum_mat[:, coord[1, 0], coord[0, 1]]
    area_2 = sum_mat[:, coord[0, 0], coord[1, 1]]
    area = area_3 - area_1 - area_2 + area_0
    return area

def get_HARR_features(sum_mat, cut_mat):
    '''
    Get HAAR features for the MNIST dataset
    '''
    fea_mat = np.zeros(shape = (sum_mat.shape[0], cut_mat.shape[0] * 2))
    for i in range(cut_mat.shape[0]):
        fea_mat[:, i] = (get_cut_area(sum_mat, cut_mat[i][[0, 2], :]) -
                         get_cut_area(sum_mat, cut_mat[i][[3, 1], :]))
        j = i + cut_mat.shape[0]
        fea_mat[:, j] = (get_cut_area(sum_mat, cut_mat[i][[0, 4], :]) -
                         get_cut_area(sum_mat, cut_mat[i][[5, 1], :]))
    return fea_mat

def main():
    print("## Load and normalize the MNIST Dataset\n")
    train, train_labels, test, test_labels = parse_MNIST(CWD + "/mnist.npz")
    train, test = edit_normalize_MNIST(train), edit_normalize_MNIST(test)
    print("## Run L2-reg Logistic Regression with all features:\n")
    clf = LogisticRegression(penalty = "l2", solver = "lbfgs", tol = 1e0,
                             max_iter = 1000)
    train_reshape = train.reshape(train.shape[0], -1)
    test_reshape = test.reshape(test.shape[0], -1)
    clf.fit(train_reshape, train_labels)
    score = clf.score(test_reshape, test_labels)
    print(f"# Purity: {score}\n")
    print("## Run L2-reg Logistic Regression with HARR features:\n")
    cut_mat = get_random_cut(28, 100, min_area = 150)
    sum_mat_train = get_sum_matrix(train, 28)
    sum_mat_test = get_sum_matrix(test, 28)
    fea_mat_train = get_HARR_features(sum_mat_train, cut_mat)
    fea_mat_test = get_HARR_features(sum_mat_test, cut_mat)
    clf.fit(fea_mat_train, train_labels)
    score_harr = clf.score(fea_mat_test, test_labels)
    print(f"# Purity: {score_harr}\n")


if __name__ == "__main__":
    main()
