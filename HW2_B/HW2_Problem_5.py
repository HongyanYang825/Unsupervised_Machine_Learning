'''
    DS 5230
    Summer 2022
    HW2_Problem_5_DBSCAN_on_toy_neighborhood_data

    Train DBSCAN clustering on toy_neighborhood_data

    Hongyan Yang
'''


import os
import pandas as pd
import matplotlib

from matplotlib import pyplot as plt

CWD = os.getcwd()

def get_num_neighbors_dict(dbscan_dict):
    '''
    Create a dictionary of pts with different # of neighbors
    '''    
    num_neighbors_dict = dict()
    for key in dbscan_dict:
        num_neighbors = len(dbscan_dict[key])
        if num_neighbors in num_neighbors_dict:
            num_neighbors_dict[num_neighbors].append(key)
        else:
            num_neighbors_dict[num_neighbors] = [key]
    return num_neighbors_dict

def dbscan_fit(dbscan_dict, num_neighbors_dict, MinPts):
    '''
    Apply DBSCAN to separate and assign clusters
    '''    
    cores, non_cores, outliers = list(), list(), list()
    [cores.extend(num_neighbors_dict[i])
     for i in num_neighbors_dict if i >= MinPts]
    [non_cores.extend(num_neighbors_dict[i])
     for i in num_neighbors_dict if i < MinPts and i > 1]
    [outliers.extend(num_neighbors_dict[i])
     for i in num_neighbors_dict if i == 1]
    # Separate and assign clusters on core points
    clusters_dict, i = dict(), 0
    while len(cores) != 0:
        cluster_set = set([each for each in dbscan_dict[cores[0]]
                           if each in cores])
        clusters_dict[i] = ([cores[0]], cluster_set)
        neighbors = list(clusters_dict[i][1] - set(clusters_dict[i][0]))
        cores.remove(cores[0])
        while len(neighbors) != 0:
            for neighbor in neighbors:
                clusters_dict[i][0].append(neighbor)
                cluster_set = set([each for each in dbscan_dict[neighbor]
                                   if each in cores])
                clusters_dict[i][1].update(cluster_set)
                cores.remove(neighbor)
            neighbors = list(clusters_dict[i][1] - set(clusters_dict[i][0]))
        i += 1
    # Separate and assign clusters on noncore points    
    non_core_list = list()
    for pt in non_cores:
        for cluster in clusters_dict:
            if not set(dbscan_dict[pt]).isdisjoint(clusters_dict[cluster][1]):
                clusters_dict[cluster][0].append(pt)
                clusters_dict[cluster][1].update(set(dbscan_dict[pt]))
                non_core_list.append(pt)
                break
    all_outliers = (set(non_cores) - set(non_core_list)).union(set(outliers))
    # Assign outliers points if exist
    if len(all_outliers) != 0:
        clusters_dict[-1] = (list(all_outliers), all_outliers)    
    return clusters_dict

def get_cluster(clusters_dict):
    '''
    Create a reverse dictionary of clusters_dict
    '''
    pt_cluster_dict = dict()
    for key in clusters_dict:
        for pt in clusters_dict[key][0]:
            pt_cluster_dict[pt] = key
    return pt_cluster_dict

def dbscan_plot(dbscan_df, col_name_1, col_name_2, k):
    '''
    Apply DBSCAN to visualize dataset
    '''
    plt.style.use("_mpl-gallery")
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(0, k)), k)
    dbscan_df.plot.scatter(x = col_name_1, y = col_name_2,
                           c = "cluster",cmap = cm)
    plt.show()

def main():
    print("### A) Load and parse the dbscan.csv file\n")
    dbscan_df = pd.DataFrame(pd.read_csv(CWD + "/dbscan.csv"))
    print(dbscan_df.head())
    neighbors_list = [[int(item) for item in each.split(",")]
                      for each in dbscan_df["neighbors"].tolist()]
    dbscan_dict = dict(zip(dbscan_df["pt"], neighbors_list))
    print()
    print("### B) Apply DBSCAN to cluster and visualize dataset\n")
    num_neighbors_dict = get_num_neighbors_dict(dbscan_dict)
    clusters_dict = dbscan_fit(dbscan_dict, num_neighbors_dict, 3)
    k = len(clusters_dict)
    pt_cluster_dict = get_cluster(clusters_dict)
    clusters_list = [pt_cluster_dict[pt] for pt in dbscan_df["pt"]]
    dbscan_df["cluster"] = clusters_list
    print("## DataFrame after DBSCAN clustering\n")
    print(dbscan_df.head())
    print()
    print("## Number of clusters with current parameters\n")
    print(k)
    dbscan_plot(dbscan_df, "x", "y", k)


if __name__ == "__main__":
    main()
