'''
    DS 5230
    Summer 2022
    HW2_Problem_6_DBSCAN_on_toy_raw_data

    Train DBSCAN clustering on three toy 2D datasets

    Hongyan Yang
'''


import math
import numpy as np

from HW2_Problem_5 import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

def parse_file(file_name, col_name_1, col_name_2):
    '''
    Load and parse the data file
    '''     
    df = pd.DataFrame(pd.read_csv(CWD + "/" + file_name))
    df["pt"] = range(len(df[col_name_1]))
    return df

def get_neighbors(vectors, epsilon, use_euclidean = True):
    '''
    Calculate each point's neighbors given epsilon
    '''
    if use_euclidean:
        distances = euclidean_distances(vectors)
    else:
        distances = cosine_similarity(vectors)
    neighbor_indices = np.argwhere(distances < epsilon)
    dbscan_dict = dict()
    for index in neighbor_indices:
        if index[0] in dbscan_dict:
            dbscan_dict[index[0]].append(index[1])
        else:
            dbscan_dict[index[0]] = [index[1]]
    return dbscan_dict

def get_dbscan_clusters(df, dbscan_dict, MinPts = 4):
    '''
    Apply DBSCAN to separate and assign clusters
    ''' 
    num_neighbors_dict = get_num_neighbors_dict(dbscan_dict)
    clusters_dict = dbscan_fit(dbscan_dict, num_neighbors_dict, MinPts)
    pt_cluster_dict = get_cluster(clusters_dict)
    clusters_list = [pt_cluster_dict[pt] for pt in df["pt"]]
    df["cluster"] = clusters_list
    k, n = len(clusters_dict), len(pt_cluster_dict)
    try:
        if -1 in clusters_dict:
            largest_component = max([len(clusters_dict[i][0]) for i in
                                     range(len(clusters_dict) - 1)]) / n
            noise = len(clusters_dict[-1][0]) / len(pt_cluster_dict)
        else:
            largest_component = max([len(clusters_dict[i][0]) for i in
                                     range(len(clusters_dict))]) / n
            noise = 0
    except:
        largest_component, noise = 0, 1
    return k, largest_component, noise

def optimize_parameters(df, vectors, MinPts = 4, initial = 0.001, iteration = 10,
                        use_euclidean = True):
    '''
    Plot share of dataset and number of clusters given different epsilons
    to choose appropriate parameter for DBSCAN
    ''' 
    epsilon_list, k_list, l_c_list, noise_list = [], [], [], []
    for i in range(iteration):
        epsilon = initial + (0.3 * i)
        dbscan_dict = get_neighbors(vectors, epsilon, use_euclidean)
        k, largest_component, noise = get_dbscan_clusters(df, dbscan_dict, MinPts)
        epsilon_list.append(epsilon)
        k_list.append(math.log(k))
        l_c_list.append(largest_component) 
        noise_list.append(noise)
    plt.style.use("_mpl-gallery")
    plt.subplot(2, 1, 1)
    plt.plot(epsilon_list, l_c_list, "go--", label = "Largest component")
    plt.plot(epsilon_list, noise_list, "rs-.", label = "Noise")
    plt.xlabel("Epsilon")
    plt.ylabel("Share of dataset")
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.plot(epsilon_list, k_list, "b^-", label = "Number of clusters")
    plt.xlabel("Epsilon")
    plt.ylabel("Number of clusters in logscale")
    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.show()

def main():
    print("### A) Load and parse the circle dataset\n")
    circle_df = parse_file("circle.csv", "Xcircle_X1", "Xcircle_X2")
    vectors = circle_df[["Xcircle_X1", "Xcircle_X2"]].to_numpy()
    print(circle_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(circle_df, vectors)
    print("Epsilon = 0.1 is appropriate for the circle dataset")
    print()
    print("## Apply DBSCAN to cluster and visualize dataset\n")
    dbscan_dict = get_neighbors(vectors, 0.1)
    k, largest_component, noise = get_dbscan_clusters(circle_df, dbscan_dict)
    print(circle_df.head())
    dbscan_plot(circle_df, "Xcircle_X1", "Xcircle_X2", k)
    print("\n")
    print("### B) Load and parse the blobs dataset\n")
    blobs_df = parse_file("blobs.csv", "Xblobs_X1", "Xblobs_X2")
    vectors = blobs_df[["Xblobs_X1", "Xblobs_X2"]].to_numpy()
    print(blobs_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(blobs_df, vectors)
    print("Epsilon = 0.5 is appropriate for the blobs dataset")
    print()
    print("## Apply DBSCAN to cluster and visualize dataset\n")
    dbscan_dict = get_neighbors(vectors, 0.5)
    k, largest_component, noise = get_dbscan_clusters(blobs_df, dbscan_dict)
    print(blobs_df.head())
    dbscan_plot(blobs_df, "Xblobs_X1", "Xblobs_X2", k)
    print("\n")
    print("### C) Load and parse the moons dataset\n")
    moons_df = parse_file("moons.csv", "Xmoons_X1", "Xmoons_X2")
    vectors = moons_df[["Xmoons_X1", "Xmoons_X2"]].to_numpy()
    print(moons_df.head())
    print()
    print("## Choose DBSCAN Parameters\n")
    #optimize_parameters(moons_df, vectors)
    print("Epsilon = 0.25 is appropriate for the moons dataset")
    print()
    print("## Apply DBSCAN to cluster and visualize dataset\n")
    dbscan_dict = get_neighbors(vectors, 0.25)
    k, largest_component, noise = get_dbscan_clusters(moons_df, dbscan_dict)
    print(moons_df.head())
    dbscan_plot(moons_df, "Xmoons_X1", "Xmoons_X2", k)


if __name__ == "__main__":
    main()
