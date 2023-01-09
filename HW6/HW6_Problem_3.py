'''
    DS 5230
    Summer 2022
    HW6_Problem_3_Social_Community_Detection

    Implement edge-removal community detection algorithm on the Flicker Graph

    Hongyan Yang
'''


import os
import itertools
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

CWD = os.getcwd()
FILE_PATH = CWD + "/Flickr_sampled_edges"

def get_betweeness(edge, paths_dict_values):
    '''
    Calculate betweeness of one edge in the graph
    '''
    betweeness = 0
    for each in paths_dict_values:
        total_paths, passed_edges = len(each), 0
        if total_paths == 0:
            continue
        for paths in each:
            if ((edge[0] in paths) and (edge[1] in paths)):
                passed_edges += 1
        betweeness += passed_edges / total_paths
    return betweeness

def remove_edge(graph, edges, paths_dict_values):
    '''
    Remove edge with largest betweeness and check graph's modularity
    '''
    betweeness = [get_betweeness(edge, paths_dict_values) for edge in edges]
    # get the indices of largest betweeness
    max_bt = max(betweeness)
    indices = [i for i, k in enumerate(betweeness) if k == max_bt]
    edges_to_remove = [edges[index] for index in indices]
    for edge in edges_to_remove:
        edges.remove(edge)
    # Construct a new graph after removing edge(s)
    order = len(np.unique(edges))
    graph = ig.Graph(order, edges)
    edges = graph.get_edgelist()
    paths_dict_keys = list(itertools.permutations(np.unique(edges), 2))
    paths_dict_values = [graph.get_all_shortest_paths(i[0], to=i[1])
                         for i in paths_dict_keys]
    clusters = graph.clusters()
    num_clusters = len(clusters)
    modularity = graph.modularity(clusters.membership)
    return graph, edges, paths_dict_values, num_clusters, modularity

def plot_graph(graph):
    '''
    Plot the graph with each cluster represented by a different color
    '''
    clusters = graph.clusters()
    num_clusters = len(clusters)
    colors = ig.drawing.colors.ClusterColoringPalette(num_clusters)
    graph.vs["color"] = colors.get_many(clusters.membership)
    ig.plot(graph)

def main():
    print("# A) Load the graph.\n")
    edges = np.loadtxt(FILE_PATH + "/edges_sampled_map_2K.csv", delimiter=",",
                       dtype="int")
    order = len(np.unique(edges))
    graph = ig.Graph(order, edges)
    edges = graph.get_edgelist()
    plot_graph(graph)
    print("# B) Remove edges until number of clusters change.\n")
    answer = input("Continue? (Y/ N) ").upper()
    while answer == "Y":
        paths_dict_keys = list(itertools.permutations(np.unique(edges), 2))
        paths_dict_values = [graph.get_all_shortest_paths(i[0], to=i[1])
                             for i in paths_dict_keys]
        clusters = graph.clusters()
        num_clusters = len(clusters)
        result = remove_edge(graph, edges, paths_dict_values)
        graph, edges, paths_dict_values = result[:3]
        new_num_clusters = result[3]
        while new_num_clusters == num_clusters:
            num_clusters = new_num_clusters
            result = remove_edge(graph, edges, paths_dict_values)
            graph, edges, paths_dict_values = result[:3]
            new_num_clusters = result[3]
        plot_graph(graph)
        answer = input("Continue? (Y/ N) ").upper()
        

if __name__ == "__main__":
    main()
