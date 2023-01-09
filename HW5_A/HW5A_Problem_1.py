'''
    DS 5230
    Summer 2022
    HW5A_Problem_1_tSNE_dimension_reduction

    Run tSNE library on MNIST and 20NG datasets and visualize the data

    Hongyan Yang
'''


import os
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category = FutureWarning)

CWD = os.getcwd()
COLORS_LIST = ["blue", "orange", "green", "red", "purple", "brown", "pink",
               "gray", "olive", "cyan"]

def parse_edit_norm_MNIST(path):
    '''
    Parse and edit normalize the MNIST dataset
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train[x_train < 50] = 0
    x_train[x_train != 0] = 1
    x_test[x_test < 50] = 0
    x_test[x_test != 0] = 1
    return x_train, y_train, x_test, y_test

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
    vct = TfidfVectorizer(decode_error = "replace", max_df = 0.95, min_df = 2,
                          stop_words = "english", max_features = max_features,
                          use_idf = use_idf)
    # Transform a list of raw texts to a matrix of tf values
    vectors = vct.fit_transform(ng_data)
    names = parse_20_NG_names(vct)
    return np.asarray(vectors.todense()), names

def plot_tsne(vectors, labels, colors_list, name, dim, perplexity,
              use_color_list = True):
    '''
    Plot data with a color per label
    '''
    if dim == 3:
        ax = plt.axes(projection = "3d")
        xdata, ydata, zdata = (vectors[:, 0], vectors[:, 1], vectors[:, 2])
        if use_color_list:
            colors = [colors_list[label] for label in labels]
            ax.scatter3D(xdata, ydata, zdata, c = colors, alpha = 0.5)
        else:
            ax.scatter3D(xdata, ydata, zdata, c = labels, alpha = 0.5)
        ax.set_title(f"{name} t_SNE data plot with perplexity = {perplexity}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        xdata, ydata = vectors[:, 0], vectors[:, 1]
        if use_color_list:
            colors = [colors_list[label] for label in labels]
            plt.scatter(xdata, ydata, c = colors, alpha = 0.5)
        else:
            plt.scatter(xdata, ydata, c = labels, alpha = 0.5)
        plt.title(f"{name} t_SNE data plot with perplexity = {perplexity}")
        plt.xlabel("x") 
        plt.ylabel("y")
    plt.show()

def main():
    print("### A) Run t_SNE on the MNIST dataset\n")
    vectors, labels = parse_edit_norm_MNIST(CWD + "/mnist.npz")[2:]
    print("## Apply PCA to reduce dimensions to 50 before conduct t_SNE\n")
    pca = PCA(n_components = 50)
    vectors = pca.fit_transform(vectors)
    print("## Reduce dimensions to 2 with t_SNE and plot the data\n")
    print("# Data plot with values for perplexity = 5")
    '''
    tsne_2_5 = TSNE(n_components=2, perplexity=5, learning_rate="auto",
                    init="pca")
    vectors_2_5 = tsne_2_5.fit_transform(vectors)
    plot_tsne(vectors_2_5[:1000], labels[:1000], COLORS_LIST, "MNIST", 2, 5)
    '''
    print("# Data plot with values for perplexity = 20")
    '''
    tsne_2_20 = TSNE(n_components=2, perplexity=20, learning_rate="auto",
                     init="pca")
    vectors_2_20 = tsne_2_20.fit_transform(vectors)
    plot_tsne(vectors_2_20[:1000], labels[:1000], COLORS_LIST, "MNIST", 2, 20)
    '''
    print("# Data plot with values for perplexity = 100")
    '''
    tsne_2_100 = TSNE(n_components=2, perplexity=100, learning_rate="auto",
                      init="pca")
    vectors_2_100 = tsne_2_100.fit_transform(vectors)
    plot_tsne(vectors_2_100[:1000], labels[:1000], COLORS_LIST, "MNIST", 2, 100)
    '''
    print()
    print("## Reduce dimensions to 3 with t_SNE and plot the data\n")
    print("# Data plot with values for perplexity = 5")
    '''
    tsne_2_5 = TSNE(n_components=3, perplexity=5, learning_rate="auto",
                    init="pca")
    vectors_2_5 = tsne_2_5.fit_transform(vectors)
    plot_tsne(vectors_2_5[:1000], labels[:1000], COLORS_LIST, "MNIST", 3, 5)
    '''
    print("# Data plot with values for perplexity = 20")
    '''
    tsne_2_20 = TSNE(n_components=3, perplexity=20, learning_rate="auto",
                     init="pca")
    vectors_2_20 = tsne_2_20.fit_transform(vectors)
    plot_tsne(vectors_2_20[:1000], labels[:1000], COLORS_LIST, "MNIST", 3, 20)
    '''
    print("# Data plot with values for perplexity = 100")
    tsne_2_100 = TSNE(n_components=3, perplexity=100, learning_rate="auto",
                      init="pca")
    vectors_2_100 = tsne_2_100.fit_transform(vectors)
    plot_tsne(vectors_2_100[:1000], labels[:1000], COLORS_LIST, "MNIST", 3, 100)
    print("\n")

    print("### B) Run t_SNE on the 20NG dataset\n")
    ng_set = fetch_20newsgroups(remove = ("headers", "footers", "quotes"),
                                subset = "test")
    vectors, names = parse_20_NG(ng_set.data, max_features = 5000,
                                 use_idf = True)
    labels = ng_set.target
    print("## Apply PCA to reduce dimensions to 50 before conduct t_SNE\n")
    pca = PCA(n_components = 50)
    vectors = pca.fit_transform(vectors)
    print("## Reduce dimensions to 2 with t_SNE and plot the data\n")
    print("# Data plot with values for perplexity = 5")
    '''
    tsne_2_5 = TSNE(n_components=2, perplexity=5, learning_rate="auto",
                    init="pca")
    vectors_2_5 = tsne_2_5.fit_transform(vectors)
    plot_tsne(vectors_2_5[:1000], labels[:1000], COLORS_LIST, "20NG", 2, 5,
              False)
    '''
    print("# Data plot with values for perplexity = 20")
    '''
    tsne_2_20 = TSNE(n_components=2, perplexity=20, learning_rate="auto",
                     init="pca")
    vectors_2_20 = tsne_2_20.fit_transform(vectors)
    plot_tsne(vectors_2_20[:1000], labels[:1000], COLORS_LIST, "20NG", 2, 20,
              False)
    '''
    print("# Data plot with values for perplexity = 100")
    '''
    tsne_2_100 = TSNE(n_components=2, perplexity=100, learning_rate="auto",
                      init="pca")
    vectors_2_100 = tsne_2_100.fit_transform(vectors)
    plot_tsne(vectors_2_100[:1000], labels[:1000], COLORS_LIST, "20NG", 2, 100,
              False)
    '''
    print()
    print("## Reduce dimensions to 3 with t_SNE and plot the data\n")
    print("# Data plot with values for perplexity = 5")
    '''
    tsne_2_5 = TSNE(n_components=3, perplexity=5, learning_rate="auto",
                    init="pca")
    vectors_2_5 = tsne_2_5.fit_transform(vectors)
    plot_tsne(vectors_2_5[:1000], labels[:1000], COLORS_LIST, "20NG", 3, 5,
              False)
    '''
    print("# Data plot with values for perplexity = 20")
    '''
    tsne_2_20 = TSNE(n_components=3, perplexity=20, learning_rate="auto",
                     init="pca")
    vectors_2_20 = tsne_2_20.fit_transform(vectors)
    plot_tsne(vectors_2_20[:1000], labels[:1000], COLORS_LIST, "20NG", 3, 20,
              False)
    '''
    print("# Data plot with values for perplexity = 100")
    tsne_2_100 = TSNE(n_components=3, perplexity=100, learning_rate="auto",
                      init="pca")
    vectors_2_100 = tsne_2_100.fit_transform(vectors)
    plot_tsne(vectors_2_100[:1000], labels[:1000], COLORS_LIST, "20NG", 3, 100,
              False)
    print("\n")
    print("### Complete")


if __name__ == "__main__":
    main()
