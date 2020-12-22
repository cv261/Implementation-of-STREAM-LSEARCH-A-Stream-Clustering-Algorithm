import sys
import os
import numpy as np
import scipy.sparse.csgraph
from sklearn import metrics
from ast import literal_eval
from cluster import Cluster
from plot import plot_clusters

'''
The CLIQUE Algorithm finds clusters by first dividing each dimension into xi equal-width intervals
and saving those intervals where the density is greater than tau. Then, each set of two
dimensions is examined: If there are two intersecting intervals in these two dimensions and the
density in the intersection of these intervals is greater than tau, the intersection is again saved as
a cluster. This is repeated for all sets of three, four, five,. . . dimensions. After every step, adjacent
clusters are replaced by a joint cluster and finally all of the clusters are output.

DDU - Dim Dense Units 
'''

# Check dimension of dense unit
def check_subdims(candidate, prev_ddu):
    for feature in candidate:
        projection = candidate.copy()
        projection.pop(feature)
        if not prev_ddu.__contains__(projection):
            return False
    return True

# Prune candidates having a (k-1) dimensional projection not in (k-1) dim dense units
def prune(candidates, prev_ddu):
    for c in candidates:
        if not check_subdims(c, prev_ddu):
            candidates.remove(c)

# If dimensionality first, insert joined items into candidate list
def insert_items(candidates, item, item2, current_dim):
    joined = item.copy()
    joined.update(item2)
    if (len(joined.keys()) == current_dim) & (not candidates.__contains__(joined)):
        candidates.append(joined)

# Check if data is in projection
def check_projection(tuple, candidate, xi):
    for feature_index, range_index in candidate.items():
        feature_value = tuple[feature_index]
        if int(feature_value * xi % xi) != range_index:
            return False
    return True

# Return candidates list
def create_candidates(prev_ddu, dim):
    candidates = []
    for i in range(len(prev_ddu)):
        for j in range(i + 1, len(prev_ddu)):
            insert_items(candidates, prev_ddu[i], prev_ddu[j], dim)
    return candidates

# Return all dense elements from candidates list
def fetch_dense_units(data, prev_ddu, dim, xi, tau):
    candidates = create_candidates(prev_ddu, dim)
    prune(candidates, prev_ddu)

    # Count number of elements in candidates
    projection = np.zeros(len(candidates))
    num_data_pts = np.shape(data)[0]
    for dataIndex in range(num_data_pts):
        for i in range(len(candidates)):
            if check_projection(data[dataIndex], candidates[i], xi):
                projection[i] += 1
    print("Projection: ", projection)

    # Return elements above density threshold
    is_dense = projection > tau * num_data_pts
    print("Dense element: ", is_dense)
    return np.array(candidates)[is_dense]

# Get edges for dense unit graph
def get_edge(node1, node2):
    dim = len(node1)
    distance = 0
    if node1.keys() != node2.keys():
        return 0
    for feature in node1.keys():
        distance += abs(node1[feature] - node2[feature])
        if distance > 1:
            return 0
    return 1

# Creates graph of dense units
def create_graph(dense_units):
    graph = np.identity(len(dense_units))
    for i in range(len(dense_units)):
        for j in range(len(dense_units)):
            graph[i, j] = get_edge(dense_units[i], dense_units[j])
    return graph

# Returns cluster data point IDs
def get_ids(data, cluster_dense_units, xi):
    point_ids = set()
    # Loop through all dense unit
    for u in cluster_dense_units:
        tmp_ids = set(range(np.shape(data)[0]))
        # Loop through all dimensions of dense unit
        for feature_index, range_index in u.items():
            tmp_ids = tmp_ids & set(
                np.where(np.floor(data[:, feature_index] * xi % xi) == range_index)[0])
        point_ids = point_ids | tmp_ids
    return point_ids

def get_clusters(dense_units, data, xi):
    graph = create_graph(dense_units)
    num_comp, comp_list = scipy.sparse.csgraph.connected_components(graph, directed=False)
    dense_units = np.array(dense_units)
    clusters = []
    
    for i in range(num_comp):
        # Get dense units of the cluster
        cluster_dense_units = dense_units[np.where(comp_list == i)]
        print("Dense units in cluster: ", cluster_dense_units.tolist())

        # Get dimensions of the cluster
        dimensions = set()
        for u in cluster_dense_units:
            dimensions.update(u.keys())

        # Get points of the cluster
        cluster_data_point_ids = get_ids(data, cluster_dense_units, xi)
        # Add cluster to list
        clusters.append(Cluster(cluster_dense_units, dimensions, cluster_data_point_ids))

    return clusters

# Evaluate performance of clustering based on a few metrics
def performance(clusters, labels):
    dim_set = set()
    for cluster in clusters:
        dim_set.add(frozenset(cluster.dimensions))

    for dim in dim_set:
        print("\nEvaluating clusters in dimension: ", list(dim))
        # Find clusters with same dimensions
        clusters_in_dim = []
        for c in clusters:
            if c.dimensions == dim:
                clusters_in_dim.append(c)
        clustering_labels = np.zeros(np.shape(labels))
        for i, c in enumerate(clusters_in_dim):
            clustering_labels[list(c.data_point_ids)] = i + 1

        print("Number of clusters: ", len(clusters_in_dim))
        print("Mutual Information Score: ", metrics.adjusted_mutual_info_score(labels, clustering_labels))
        print("Homogeneity, Completeness, V-measure: ", metrics.homogeneity_completeness_v_measure(labels, clustering_labels))

# Returns dense units of dim 1
def get_one_ddu(data, tau, xi):
    num_data_pts = np.shape(data)[0]
    num_feat = np.shape(data)[1]
    projection = np.zeros((xi, num_feat))
    for f in range(num_feat):
        for element in data[:, f]:
            projection[int(element * xi % xi), f] += 1
    print("1D projection:\n", projection, "\n")
    is_dense = projection > tau * num_data_pts
    print("is_dense:\n", is_dense)
    one_ddu = []
    for f in range(num_feat):
        for unit in range(xi):
            if is_dense[unit, f]:
                dense_unit = dict({f: unit})
                one_ddu.append(dense_unit)
    return one_ddu

# Run CLIQUE algorithm
def clique(data, xi, tau):
    # Finding 1 dimensional dense units
    dense_units = get_one_ddu(data, tau, xi)

    # Getting 1 dimensional clusters
    clusters = get_clusters(dense_units, data, xi)

    # Finding dense units and clusters for dimension > 2
    current_dim = 2
    num_feat = np.shape(data)[1]
    while (current_dim <= num_feat) & (len(dense_units) > 0):
        print("\n", str(current_dim), " dimensional clusters:")
        dense_units = fetch_dense_units(
            data, dense_units, current_dim, xi, tau)
        for cluster in get_clusters(dense_units, data, xi):
            clusters.append(cluster)
        current_dim += 1
    return clusters

# Read labels from dataset
def get_labels(delimiter, label_column, path):
    return np.genfromtxt(path, dtype="U10", delimiter=delimiter, usecols=[label_column])

# Read data from dataset
def get_data(delimiter, feature_columns, path):
    return np.genfromtxt(path, dtype=float, delimiter=delimiter, usecols=feature_columns)

# Save cluster info in a txt file
def save_info(clusters, ds_file):
    file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), ds_file), encoding='utf-8', mode="w+")
    for i, c in enumerate(clusters):
        c.id = i
        file.write("Cluster " + str(i) + ":\n" + str(c))
    file.close()

# Normalize data in all features (1e-5 padding is added because clustering works on [0,1) interval)
def normalize_features(data):
    normalized_data = data
    num_feat = np.shape(normalized_data)[1]
    for f in range(num_feat):
        normalized_data[:, f] -= min(normalized_data[:, f]) - 1e-5
        normalized_data[:, f] *= 1 / (max(normalized_data[:, f]) + 1e-5)
    return normalized_data

def visualize_clusters(features, data, clusters, title, xi, tau, file)
    title = ("Dataset: " + file + "tau = " +str(tau) + "xi = " + str(xi))
    if len(features) <= 2:
        plot_clusters(data, clusters, title, xi)

if __name__ == "__main__":
    xi = 3
    tau = 0.1
    ds_file = "data.txt"
    feature_columns = [4, 5, 6, 7]
    label_column = 3
    delimiter = ' '
    cluster_file = "clusters_info.txt"
    
    # ds_file = "mouse.csv"
    # feature_columns = [0, 1]
    # label_column = 2
    # xi = 3
    # tau = 0.1
    # delimiter = ' '
    # cluster_file = "output_clusters.txt"

    print("Running CLIQUE on " + ds_file + " dataset | xi = " + str(xi) + " | tau = " + str(tau) + "\n")

    # Read data
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), ds_file)
    data = get_data(delimiter, feature_columns, path)
    labels = get_labels(delimiter, label_column, path)

    # Normalize data to [0,1]
    norm_data = normalize_features(data)

    # Run CLIQUE on dataset
    clusters = clique(data=norm_data, xi=xi, tau=tau)
    # Save clusters to txt file
    save_info(clusters, cluster_file)
    print("\nClusters saved to " + cluster_file)

    # Evaluate results
    performance(clusters, labels)

    # Visualize clusters
    visualize_clusters(feature_columns, norm_data, clusters, title, xi, tau, ds_file)