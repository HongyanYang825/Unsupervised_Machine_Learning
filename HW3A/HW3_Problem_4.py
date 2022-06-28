'''
    DS 5230
    Summer 2022
    HW3A_Problem_4_PCA_for_cluster_visualization

    Plot data in 3D with PCA representation

    Hongyan Yang
'''


import matplotlib.pyplot as plt

from HW3_Problem_1 import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from statistics import mode

COLORS_LIST = ["blue", "orange", "green", "red", "purple", "brown",
               "pink", "gray", "olive", "cyan"]

def get_true_labels(predicted_labels, labels, k):
    '''
    Get the "true" labels of predicted_labels
    '''
    labels_dict = dict()
    for i in range(k):
        labels_dict[i] = np.where(predicted_labels == i)[0]
    updated_labels = np.empty(predicted_labels.shape)
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

def designs_mapping(labels, k, designs_list):
    '''
    Map data in each cluster with different marker or color
    '''
    labels_dict = dict()
    for i in range(k):
        labels_dict[i] = np.where(labels == i)[0]
    designs = np.empty(labels.shape, dtype = np.object_)
    for i in range(k):
        designs[labels_dict[i]] = designs_list[i]
    return designs

def plot_pca(vectors, predicted_labels, labels, markers, colors):
    '''
    Plot data in 3D with given 3 eigen values
    '''
    ax = plt.axes(projection = "3d")
    xdata, ydata, zdata = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    for x, y, z, m, c in zip(xdata, ydata, zdata, markers, colors):
        ax.scatter3D(x, y, z, marker = m, c = c)
    ax.set_title("MNIST data plot with given eigen vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def main():
    print("### A) Run KMeans on the MNIST dataset\n")
    mnist_data = parse_MNIST(CWD + "/mnist.npz")
    vectors = edit_normalize_MNIST(mnist_data[2])
    labels = mnist_data[3]
    kmeans = KMeans(n_clusters = 10, n_init = 10).fit(vectors)
    predicted_labels = get_true_labels(kmeans.labels_, labels, 10)
    purity = round(sum(predicted_labels == labels) / len(labels), 4)
    gini_index = get_gini_index(predicted_labels, labels, 10)
    print(f"## Purity: {purity}")
    print(f"## Gini Index: {gini_index}")
    print("\n")
    print("### B) Run PCA on the MNIST dataset\n")
    pca = PCA(n_components = 20)
    vectors_pca = pca.fit_transform(vectors)
    kmeans = KMeans(n_clusters = 10, n_init = 10).fit(vectors_pca)
    predicted_labels = get_true_labels(kmeans.labels_, labels, 10)
    purity = round(sum(predicted_labels == labels) / len(labels), 4)
    gini_index = get_gini_index(predicted_labels, labels, 10)
    print(f"## Purity: {purity}")
    print(f"## Gini Index: {gini_index}")
    print("\n")
    print("### C) Plot data in 3D with top 3 eigen values\n")
    vectors_plot, labels_plot = vectors_pca[:300], labels[:300]
    predicted_labels_plot = predicted_labels[:300]
    markers_list = list(Line2D.markers.keys())
    markers = designs_mapping(labels_plot, 10, markers_list)
    colors = designs_mapping(predicted_labels_plot, 10, COLORS_LIST)
    plot_pca(vectors_plot, predicted_labels_plot,
             labels_plot, markers, colors)
    print()
    print("### D) Plot data in 3D with other 3 eigen values\n")
    replot = "Y"
    while replot == "Y":
        indices = np.random.choice(20, 3, replace = False)
        vectors_random = vectors_plot[:, indices]
        plot_pca(vectors_random, predicted_labels_plot,
                 labels_plot, markers, colors)
        replot = input("Would you like to replot (Y/ N)? ").upper()
        print()


if __name__ == "__main__":
    main()
