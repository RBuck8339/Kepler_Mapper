"""

Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips.

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_

"""

import random

import io
import sys
import base64

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import kmapper as km
import pandas as pd
import json

import torch
# from rdkit.Chem.rdmolfiles import MolFromSmiles
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder  # Given at start
from ogb.utils import features  # To get my features
from ogb.utils import mol  # To make the SMILES into a tensor
from ogb.utils import torch_util  # If needed, to create numpy to tensor
from ogb.graphproppred import PygGraphPropPredDataset  # To get my dataset to work with

from rdkit import Chem
from torch_geometric.utils import to_dense_adj
import networkx as nx
from torch_geometric.data import Data

from sklearn.model_selection import GridSearchCV

from helpers import createFolder
from helpers import copyFile

dir = 'C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/'
html_template = dir + "Outputs/Visualizations/digits_custom_tooltips.html"

random.seed(42)
'''
Notes:

Node Embeddings will not work for the purposes of this, the program crashes early due to there being too many data points

Birch Clusterer seems to work best as it captures all embeddings
DBSCAN Clusterer removes many embeddings

Trying to lower embedding dimensions for nodes and seeing if this works better
DBSCAN results in memory error
Birch results in poor visualization
'''

# Given at start
atom_encoder = AtomEncoder(emb_dim=60)  # Class object for node embeddings
# These automatically scale the data with an Xavier Uniform Distribution

dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='/OGB_Dataset')  # Load the dataset
print(dataset[0])
print(len(dataset))
# Generate labels for graph

labels = pd.read_csv(dir + 'OGB_Dataset/ogbg_molhiv/mapping/mol.csv.gz', compression='gzip').iloc[:,
         0]  # Get the graph labels
labels = labels.to_list()

total_ones = labels.count(1)
total_zeros = labels.count(0)

print("Number of 1s:", total_ones)  # display number of ones in the dataset
print("Number of 0s:", total_zeros)  # display number of zeros in the dataset

# Generate labels for visualization
binary_labels = []

for i in range(len(dataset)):
    for label in range(dataset[i].num_nodes):
        binary_labels.append(labels[label])

tooltip_s = np.array(binary_labels)

# Generate embeddings for all nodes in dataset
tlist_atom_embeddings = []
nplist_atom_embeddings = []
for node in range(len(dataset)):
    # Get the embeddings for current SMILES
    atom_embedding = atom_encoder(dataset[node].x)  # Get the embedding of given SMILES equation
    tlist_atom_embeddings.append(atom_embedding)  # Add tensor to our list of tensors

for tensor in tlist_atom_embeddings:
    for item in tensor:
        nplist_atom_embeddings.append(item.detach().numpy())

nplist_atom_embeddings = np.asarray(nplist_atom_embeddings)

# Initialize the mapper object
mapper = km.KeplerMapper(verbose=2)

'''
KMeans = sklearn.cluster.KMeans()

n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
projection = [sklearn.manifold.TSNE(), sklearn.manifold.SpectralEmbedding(), sklearn.manifold.MDS()]
random_state = [20, 42, 60, 80, 100]
n_init = [1, 2, 3, 4, 5, 6, 7]

random_grid = {'n_clusters': n_clusters,
               'random_state': random_state,
               'n_init': n_init}

# Fit and transform data
grid_result = GridSearchCV(estimator=KMeans, param_grid=random_grid, cv=3, verbose=2, n_jobs=-1)
'''

projected_data = mapper.fit_transform(nplist_atom_embeddings, projection=sklearn.manifold.TSNE(), scaler=None)

folder_dir = dir + "ForKiarash"
createFolder(folder_dir)
createFolder(folder_dir + "/Visualizations")
copyFile(folder_dir + "/Visualizations", html_template)
folder_dir = folder_dir + "/"

graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.KMeans(n_clusters=10, random_state=42, n_init=1),
    # clusterer=grid_result,
    cover=km.Cover(10, 0.6))
# cover=km.Cover(n_cubes=5, perc_overlap=0.4))
# Creat a networkX graph for TDA mapper graph, in this graph nodes will be the clusters and the node featre would be the cluster size

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
# print("Output graph examples to html")

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html=folder_dir + "Visualizations/digits_custom_tooltips.html",
    color_values=binary_labels,
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)

num_members = []
for key in graph["nodes"]:
    num_members.append(len(graph["nodes"][key]))

num_members_df = pd.DataFrame(num_members)
num_members_df.index = graph["nodes"].keys()
num_members_df = num_members_df.transpose()
num_members_df.to_csv(folder_dir + 'NumMembers.csv', index=False)

print("Wrote dataframe containing number of members in each cluster to: Outputs/NumMembers.csv")

'''
count_nodes_keys = graph["nodes"].keys()
members_df = pd.DataFrame(graph_memberships, columns=count_nodes_keys)
members_df.to_csv()
'''

count_graph_nodes_dict = dict.fromkeys(graph["nodes"].keys())

list_sizes = ["0-5 nodes", "6-15 nodes", "16-30 nodes", "31-60 nodes", "61-100 nodes", "101-250 nodes", "251-500 nodes",
              "501-1000 nodes", "1001-2000 nodes",
              "2001-4000 nodes", "4001-6000 nodes", "6001-10000 nodes", "10001-15000 nodes", "15001-20000 nodes",
              "20001-30000 nodes",
              "30001-45000 nodes", "45001-60000 nodes", "60001-80000 nodes", "80001-100000 nodes",
              "100001-150000 nodes", "150001+ nodes"]

list_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for key in list(graph["nodes"].keys()):
    if len(graph["nodes"][key]) > 0 and len(graph["nodes"][key]) <= 5:
        list_vals[0] = list_vals[0] + 1
    elif len(graph["nodes"][key]) > 5 and len(graph["nodes"][key]) <= 15:
        list_vals[1] = list_vals[1] + 1
    elif len(graph["nodes"][key]) > 15 and len(graph["nodes"][key]) <= 30:
        list_vals[2] = list_vals[2] + 1
    elif len(graph["nodes"][key]) > 30 and len(graph["nodes"][key]) <= 60:
        list_vals[3] = list_vals[3] + 1
    elif len(graph["nodes"][key]) > 60 and len(graph["nodes"][key]) <= 100:
        list_vals[4] = list_vals[4] + 1
    elif len(graph["nodes"][key]) > 100 and len(graph["nodes"][key]) <= 250:
        list_vals[5] = list_vals[5] + 1
    elif len(graph["nodes"][key]) > 250 and len(graph["nodes"][key]) <= 500:
        list_vals[6] = list_vals[6] + 1
    elif len(graph["nodes"][key]) > 500 and len(graph["nodes"][key]) <= 1000:
        list_vals[7] = list_vals[7] + 1
    elif len(graph["nodes"][key]) > 1000 and len(graph["nodes"][key]) <= 2000:
        list_vals[8] = list_vals[8] + 1
    elif len(graph["nodes"][key]) > 2000 and len(graph["nodes"][key]) <= 4000:
        list_vals[9] = list_vals[9] + 1
    elif len(graph["nodes"][key]) > 4000 and len(graph["nodes"][key]) <= 6000:
        list_vals[10] = list_vals[10] + 1
    elif len(graph["nodes"][key]) > 6000 and len(graph["nodes"][key]) <= 10000:
        list_vals[11] = list_vals[11] + 1
    elif len(graph["nodes"][key]) > 10000 and len(graph["nodes"][key]) <= 15000:
        list_vals[12] = list_vals[12] + 1
    elif len(graph["nodes"][key]) > 15000 and len(graph["nodes"][key]) <= 20000:
        list_vals[13] = list_vals[13] + 1
    elif len(graph["nodes"][key]) > 20000 and len(graph["nodes"][key]) <= 30000:
        list_vals[14] = list_vals[14] + 1
    elif len(graph["nodes"][key]) > 30000 and len(graph["nodes"][key]) <= 45000:
        list_vals[15] = list_vals[15] + 1
    elif len(graph["nodes"][key]) > 45000 and len(graph["nodes"][key]) <= 60000:
        list_vals[16] = list_vals[16] + 1
    elif len(graph["nodes"][key]) > 60000 and len(graph["nodes"][key]) <= 80000:
        list_vals[17] = list_vals[17] + 1
    elif len(graph["nodes"][key]) > 80000 and len(graph["nodes"][key]) <= 100000:
        list_vals[18] = list_vals[18] + 1
    elif len(graph["nodes"][key]) > 100000 and len(graph["nodes"][key]) <= 150000:
        list_vals[19] = list_vals[19] + 1
    elif len(graph["nodes"][key]) > 150000:
        list_vals[20] = list_vals[20] + 1

count_node_sizes_dict = dict(zip(list_sizes, list_vals))

index = ["num_occurrances"]
count_graph_nodes_df = pd.DataFrame(count_node_sizes_dict, index=index)

count_graph_nodes_df.to_csv(folder_dir + 'ClusterSizes.csv', index=False)

print("Wrote dataframe containing cluster sizes to " + folder_dir + "ClusterSizes.csv")

list_sizes = ["1-5 nodes", "6-9 nodes", "10-12 nodes", "13-14 nodes", "15 nodes", "16 nodes", "17 nodes", "18 nodes",
              "19 nodes", "20 nodes", "21 nodes", "22 nodes",
              "23 nodes", "24 nodes", "25 nodes", "26 nodes", "27 nodes", "28 nodes", "29 nodes", "30 nodes",
              "31 nodes", "32 nodes", "33 nodes", "34 nodes", "35 nodes",
              "36-38 nodes", "39-40 nodes", "41-43 nodes", "44-45 nodes", "46+ nodes"]

list_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(dataset)):
    if dataset[i].num_nodes > 0 and dataset[i].num_nodes <= 5:
        list_vals[0] = list_vals[0] + 1
    elif dataset[i].num_nodes > 5 and dataset[i].num_nodes <= 9:
        list_vals[1] = list_vals[1] + 1
    elif dataset[i].num_nodes > 9 and dataset[i].num_nodes <= 12:
        list_vals[2] = list_vals[2] + 1
    elif dataset[i].num_nodes > 12 and dataset[i].num_nodes <= 14:
        list_vals[3] = list_vals[3] + 1
    elif dataset[i].num_nodes == 15:
        list_vals[4] = list_vals[4] + 1
    elif dataset[i].num_nodes == 16:
        list_vals[5] = list_vals[5] + 1
    elif dataset[i].num_nodes == 17:
        list_vals[6] = list_vals[6] + 1
    elif dataset[i].num_nodes == 18:
        list_vals[7] = list_vals[7] + 1
    elif dataset[i].num_nodes == 19:
        list_vals[8] = list_vals[8] + 1
    elif dataset[i].num_nodes == 20:
        list_vals[9] = list_vals[9] + 1
    elif dataset[i].num_nodes == 21:
        list_vals[10] = list_vals[10] + 1
    elif dataset[i].num_nodes == 22:
        list_vals[11] = list_vals[11] + 1
    elif dataset[i].num_nodes == 23:
        list_vals[12] = list_vals[12] + 1
    elif dataset[i].num_nodes == 24:
        list_vals[13] = list_vals[13] + 1
    elif dataset[i].num_nodes == 25:
        list_vals[14] = list_vals[14] + 1
    elif dataset[i].num_nodes == 26:
        list_vals[15] = list_vals[15] + 1
    elif dataset[i].num_nodes == 27:
        list_vals[16] = list_vals[16] + 1
    elif dataset[i].num_nodes == 28:
        list_vals[17] = list_vals[17] + 1
    elif dataset[i].num_nodes == 29:
        list_vals[18] = list_vals[18] + 1
    elif dataset[i].num_nodes == 30:
        list_vals[19] = list_vals[19] + 1
    elif dataset[i].num_nodes == 31:
        list_vals[20] = list_vals[20] + 1
    elif dataset[i].num_nodes == 32:
        list_vals[21] = list_vals[21] + 1
    elif dataset[i].num_nodes == 33:
        list_vals[22] = list_vals[22] + 1
    elif dataset[i].num_nodes == 34:
        list_vals[23] = list_vals[23] + 1
    elif dataset[i].num_nodes == 35:
        list_vals[24] = list_vals[24] + 1
    elif dataset[i].num_nodes > 35 and dataset[i].num_nodes <= 38:
        list_vals[25] = list_vals[25] + 1
    elif dataset[i].num_nodes > 39 and dataset[i].num_nodes <= 40:
        list_vals[27] = list_vals[26] + 1
    elif dataset[i].num_nodes > 40 and dataset[i].num_nodes <= 43:
        list_vals[28] = list_vals[27] + 1
    elif dataset[i].num_nodes > 43 and dataset[i].num_nodes <= 45:
        list_vals[29] = list_vals[28] + 1
    elif dataset[i].num_nodes > 45:
        list_vals[30] = list_vals[29] + 1

count_node_sizes_dict = dict(zip(list_sizes, list_vals))

index = ["num_occurrances"]
count_graph_nodes_df = pd.DataFrame(count_node_sizes_dict, index=index)

count_graph_nodes_df.to_csv(folder_dir + 'GraphSizes.csv', index=False)

print("Wrote dataframe containing graph sizes to " + folder_dir + "GraphSizes.csv")

graph_copy = graph

graph_indicators = []
with open('OGB_Dataset\Created_Files\Graph_Indicator.txt', "r") as f:
    for line in f:
        graph_indicators.append(int(line))

# Get all column names
columns = ["graph number"]
columns = columns + list(graph["nodes"].keys())
columns = columns + ["Graph Label"]

# Modify the graph_copy to have graph indicators so we can count where the nodes appear
for key, values in graph_copy["nodes"].items():
    for i in range(len(values)):
        values[i] = graph_indicators[values[i] - 1]

num_nodes_cluster_df = pd.DataFrame(columns=columns)

for num in range(len(dataset)):
    new_row = []
    new_row.append(num)
    for key in graph_copy["nodes"].keys():
        new_row.append((graph_copy["nodes"][key]).count(num))
    new_row.append(dataset[num].y)
    num_nodes_cluster_df.loc[len(num_nodes_cluster_df)] = new_row

num_nodes_cluster_df.to_csv(folder_dir + 'GraphNodeDistributions.csv', index=False)

print("Wrote dataframe containing node distributions to " + folder_dir + "GraphNodeDistributions.csv")


new_graph = graph  # Reset graph_copy

# Prints out all nodes in their clusters with their node labels, what I think I will want to do is, for each node in node,
#  then give them their graph label from the txt file

for key in (new_graph["nodes"]):
    for value in range(len(list(new_graph["nodes"][key]))):
        new_graph["nodes"][key][value] = str(binary_labels[new_graph["nodes"][key][value]])

total_ones = 0
total_zeros = 0

for key in new_graph["nodes"]:
    total_ones += new_graph["nodes"][key].count("1")
    total_zeros += new_graph["nodes"][key].count("0")

list_nodes_info = []

for key in new_graph["nodes"]:
    # print(key, " size is: ", len(graph_copy["nodes"][key]))
    ones_in_cluster = new_graph["nodes"][key].count("1")
    zeros_in_cluster = new_graph["nodes"][key].count("0")

    ones_not_in_cluster = total_ones - ones_in_cluster
    zeros_not_in_cluster = total_zeros - zeros_in_cluster

    if (zeros_in_cluster != 0):
        odds_ratio = (ones_in_cluster / zeros_in_cluster) / (ones_not_in_cluster / zeros_not_in_cluster)
    else:
        odds_ratio = 100

    # print("Odds Ratio is: ", odds_ratio)

    list_nodes_info.append((key, ones_in_cluster, zeros_in_cluster, odds_ratio))

# Define column names (optional, but recommended)
columns = ["key", "number of ones", "number of zeros", "odds ratio"]

# Create DataFrame from list of tuples
nodes_df = pd.DataFrame(list_nodes_info, columns=columns)
nodes_df = nodes_df.sort_values(by='odds ratio', ascending=False)
nodes_df.to_csv(folder_dir + 'OddsRatio.csv')

print("Wrote Odds Ratios to: " + folder_dir + "OddsRatio.csv")

for i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
    for j in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        for k in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

            folder_dir = dir + "Outputs/ParamSearch/nclusters" + str(i) + "_ncubes" + str(j) + "_percoverlap" + str(k)

            createFolder(folder_dir)

            folder_dir = folder_dir + "/"

            createFolder(folder_dir + "Visualizations")
            copyFile(folder_dir + "Visualizations", html_template)

            graph = mapper.map(
                projected_data,
                clusterer=sklearn.cluster.KMeans(n_clusters=i, random_state=42, n_init=1),
                # clusterer=grid_result,
                cover=km.Cover(j, k))
            # cover=km.Cover(n_cubes=5, perc_overlap=0.4))

            # Tooltips with image data for every cluster member
            mapper.visualize(
                graph,
                title="Handwritten digits Mapper",
                path_html=folder_dir + "Visualizations/digits_custom_tooltips.html",
                color_values=binary_labels,
                color_function_name="labels",
                custom_tooltips=tooltip_s,
            )

            num_members = []
            for key in graph["nodes"]:
                num_members.append(len(graph["nodes"][key]))

            num_members_df = pd.DataFrame(num_members)
            num_members_df.index = graph["nodes"].keys()
            num_members_df = num_members_df.transpose()
            num_members_df.to_csv(folder_dir + 'NumMembers.csv', index=False)

            print("Wrote dataframe containing number of members in each cluster to: " + folder_dir + "NumMembers.csv")

            '''
            count_nodes_keys = graph["nodes"].keys()
            members_df = pd.DataFrame(graph_memberships, columns=count_nodes_keys)
            members_df.to_csv()
            '''

            count_graph_nodes_dict = dict.fromkeys(graph["nodes"].keys())

            list_sizes = ["0-5 nodes", "6-15 nodes", "16-30 nodes", "31-60 nodes", "61-100 nodes", "101-250 nodes",
                          "251-500 nodes", "501-1000 nodes", "1001-2000 nodes",
                          "2001-4000 nodes", "4001-6000 nodes", "6001-10000 nodes", "10001-15000 nodes",
                          "15001-20000 nodes", "20001-30000 nodes",
                          "30001-45000 nodes", "45001-60000 nodes", "60001-80000 nodes", "80001-100000 nodes",
                          "100001-150000 nodes", "150001+ nodes"]

            list_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for key in list(graph["nodes"].keys()):
                if len(graph["nodes"][key]) > 0 and len(graph["nodes"][key]) <= 5:
                    list_vals[0] = list_vals[0] + 1
                elif len(graph["nodes"][key]) > 5 and len(graph["nodes"][key]) <= 15:
                    list_vals[1] = list_vals[1] + 1
                elif len(graph["nodes"][key]) > 15 and len(graph["nodes"][key]) <= 30:
                    list_vals[2] = list_vals[2] + 1
                elif len(graph["nodes"][key]) > 30 and len(graph["nodes"][key]) <= 60:
                    list_vals[3] = list_vals[3] + 1
                elif len(graph["nodes"][key]) > 60 and len(graph["nodes"][key]) <= 100:
                    list_vals[4] = list_vals[4] + 1
                elif len(graph["nodes"][key]) > 100 and len(graph["nodes"][key]) <= 250:
                    list_vals[5] = list_vals[5] + 1
                elif len(graph["nodes"][key]) > 250 and len(graph["nodes"][key]) <= 500:
                    list_vals[6] = list_vals[6] + 1
                elif len(graph["nodes"][key]) > 500 and len(graph["nodes"][key]) <= 1000:
                    list_vals[7] = list_vals[7] + 1
                elif len(graph["nodes"][key]) > 1000 and len(graph["nodes"][key]) <= 2000:
                    list_vals[8] = list_vals[8] + 1
                elif len(graph["nodes"][key]) > 2000 and len(graph["nodes"][key]) <= 4000:
                    list_vals[9] = list_vals[9] + 1
                elif len(graph["nodes"][key]) > 4000 and len(graph["nodes"][key]) <= 6000:
                    list_vals[10] = list_vals[10] + 1
                elif len(graph["nodes"][key]) > 6000 and len(graph["nodes"][key]) <= 10000:
                    list_vals[11] = list_vals[11] + 1
                elif len(graph["nodes"][key]) > 10000 and len(graph["nodes"][key]) <= 15000:
                    list_vals[12] = list_vals[12] + 1
                elif len(graph["nodes"][key]) > 15000 and len(graph["nodes"][key]) <= 20000:
                    list_vals[13] = list_vals[13] + 1
                elif len(graph["nodes"][key]) > 20000 and len(graph["nodes"][key]) <= 30000:
                    list_vals[14] = list_vals[14] + 1
                elif len(graph["nodes"][key]) > 30000 and len(graph["nodes"][key]) <= 45000:
                    list_vals[15] = list_vals[15] + 1
                elif len(graph["nodes"][key]) > 45000 and len(graph["nodes"][key]) <= 60000:
                    list_vals[16] = list_vals[16] + 1
                elif len(graph["nodes"][key]) > 60000 and len(graph["nodes"][key]) <= 80000:
                    list_vals[17] = list_vals[17] + 1
                elif len(graph["nodes"][key]) > 80000 and len(graph["nodes"][key]) <= 100000:
                    list_vals[18] = list_vals[18] + 1
                elif len(graph["nodes"][key]) > 100000 and len(graph["nodes"][key]) <= 150000:
                    list_vals[19] = list_vals[19] + 1
                elif len(graph["nodes"][key]) > 150000:
                    list_vals[20] = list_vals[20] + 1

            count_node_sizes_dict = dict(zip(list_sizes, list_vals))

            index = ["num_occurrances"]
            count_graph_nodes_df = pd.DataFrame(count_node_sizes_dict, index=index)

            count_graph_nodes_df.to_csv(folder_dir + 'ClusterSizes.csv', index=False)

            print("Wrote dataframe containing cluster sizes to " + folder_dir + "ClusterSizes.csv")

            list_sizes = ["1-5 nodes", "6-9 nodes", "10-12 nodes", "13-14 nodes", "15 nodes", "16 nodes", "17 nodes",
                          "18 nodes", "19 nodes", "20 nodes", "21 nodes", "22 nodes",
                          "23 nodes", "24 nodes", "25 nodes", "26 nodes", "27 nodes", "28 nodes", "29 nodes",
                          "30 nodes", "31 nodes", "32 nodes", "33 nodes", "34 nodes", "35 nodes",
                          "36-38 nodes", "39-40 nodes", "41-43 nodes", "44-45 nodes", "46+ nodes"]

            list_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for l in range(len(dataset)):
                if dataset[l].num_nodes > 0 and dataset[l].num_nodes <= 5:
                    list_vals[0] = list_vals[0] + 1
                elif dataset[l].num_nodes > 5 and dataset[l].num_nodes <= 9:
                    list_vals[1] = list_vals[1] + 1
                elif dataset[l].num_nodes > 9 and dataset[l].num_nodes <= 12:
                    list_vals[2] = list_vals[2] + 1
                elif dataset[l].num_nodes > 12 and dataset[l].num_nodes <= 14:
                    list_vals[3] = list_vals[3] + 1
                elif dataset[l].num_nodes == 15:
                    list_vals[4] = list_vals[4] + 1
                elif dataset[l].num_nodes == 16:
                    list_vals[5] = list_vals[5] + 1
                elif dataset[l].num_nodes == 17:
                    list_vals[6] = list_vals[6] + 1
                elif dataset[l].num_nodes == 18:
                    list_vals[7] = list_vals[7] + 1
                elif dataset[l].num_nodes == 19:
                    list_vals[8] = list_vals[8] + 1
                elif dataset[l].num_nodes == 20:
                    list_vals[9] = list_vals[9] + 1
                elif dataset[l].num_nodes == 21:
                    list_vals[10] = list_vals[10] + 1
                elif dataset[l].num_nodes == 22:
                    list_vals[11] = list_vals[11] + 1
                elif dataset[l].num_nodes == 23:
                    list_vals[12] = list_vals[12] + 1
                elif dataset[l].num_nodes == 24:
                    list_vals[13] = list_vals[13] + 1
                elif dataset[l].num_nodes == 25:
                    list_vals[14] = list_vals[14] + 1
                elif dataset[l].num_nodes == 26:
                    list_vals[15] = list_vals[15] + 1
                elif dataset[l].num_nodes == 27:
                    list_vals[16] = list_vals[16] + 1
                elif dataset[i].num_nodes == 28:
                    list_vals[17] = list_vals[17] + 1
                elif dataset[l].num_nodes == 29:
                    list_vals[18] = list_vals[18] + 1
                elif dataset[l].num_nodes == 30:
                    list_vals[19] = list_vals[19] + 1
                elif dataset[l].num_nodes == 31:
                    list_vals[20] = list_vals[20] + 1
                elif dataset[l].num_nodes == 32:
                    list_vals[21] = list_vals[21] + 1
                elif dataset[l].num_nodes == 33:
                    list_vals[22] = list_vals[22] + 1
                elif dataset[l].num_nodes == 34:
                    list_vals[23] = list_vals[23] + 1
                elif dataset[l].num_nodes == 35:
                    list_vals[24] = list_vals[24] + 1
                elif dataset[l].num_nodes > 35 and dataset[l].num_nodes <= 38:
                    list_vals[25] = list_vals[25] + 1
                elif dataset[l].num_nodes > 39 and dataset[l].num_nodes <= 40:
                    list_vals[27] = list_vals[26] + 1
                elif dataset[l].num_nodes > 40 and dataset[l].num_nodes <= 43:
                    list_vals[28] = list_vals[27] + 1
                elif dataset[l].num_nodes > 43 and dataset[l].num_nodes <= 45:
                    list_vals[29] = list_vals[28] + 1
                elif dataset[l].num_nodes > 45:
                    list_vals[30] = list_vals[29] + 1

            count_node_sizes_dict = dict(zip(list_sizes, list_vals))

            index = ["num_occurrances"]
            count_graph_nodes_df = pd.DataFrame(count_node_sizes_dict, index=index)

            count_graph_nodes_df.to_csv(folder_dir + 'GraphSizes.csv', index=False)

            print("Wrote dataframe containing graph sizes to " + folder_dir + "GraphSizes.csv")

            graph_copy = graph

            graph_indicators = []
            with open('OGB_Dataset\Created_Files\Graph_Indicator.txt', "r") as f:
                for line in f:
                    graph_indicators.append(int(line))

            # Get all column names
            columns = ["graph number"]
            columns = columns + list(graph["nodes"].keys())
            columns = columns + ["Graph Label"]

            # Modify the graph_copy to have graph indicators so we can count where the nodes appear
            for key, values in graph_copy["nodes"].items():
                for i in range(len(values)):
                    values[i] = graph_indicators[values[i] - 1]

            num_nodes_cluster_df = pd.DataFrame(columns=columns)

            for num in range(len(dataset)):
                new_row = []
                new_row.append(num)
                for key in graph_copy["nodes"].keys():
                    new_row.append((graph_copy["nodes"][key]).count(num))
                new_row.append(dataset[num].y)
                num_nodes_cluster_df.loc[len(num_nodes_cluster_df)] = new_row

            num_nodes_cluster_df.to_csv(folder_dir + 'GraphNodeDistributions.csv', index=False)

            print("Worte dataframe containing node distributions to " + folder_dir + "GraphNodeDistributions.csv")

            new_graph = graph  # Reset graph_copy

            # Prints out all nodes in their clusters with their node labels, what I think I will want to do is, for each node in node,
            #  then give them their graph label from the txt file

            for key in (new_graph["nodes"]):
                for value in range(len(list(new_graph["nodes"][key]))):
                    new_graph["nodes"][key][value] = str(binary_labels[new_graph["nodes"][key][value]])

            total_ones = 0
            total_zeros = 0

            for key in new_graph["nodes"]:
                total_ones += new_graph["nodes"][key].count("1")
                total_zeros += new_graph["nodes"][key].count("0")

            list_nodes_info = []

            for key in new_graph["nodes"]:
                # print(key, " size is: ", len(graph_copy["nodes"][key]))
                ones_in_cluster = new_graph["nodes"][key].count("1")
                zeros_in_cluster = new_graph["nodes"][key].count("0")

                ones_not_in_cluster = total_ones - ones_in_cluster
                zeros_not_in_cluster = total_zeros - zeros_in_cluster

                if (zeros_in_cluster != 0):
                    odds_ratio = (ones_in_cluster / zeros_in_cluster) / (ones_not_in_cluster / zeros_not_in_cluster)
                else:
                    odds_ratio = 100

                # print("Odds Ratio is: ", odds_ratio)

                list_nodes_info.append((key, ones_in_cluster, zeros_in_cluster, odds_ratio))

            # Define column names (optional, but recommended)
            columns = ["key", "number of ones", "number of zeros", "odds ratio"]

            # Create DataFrame from list of tuples
            nodes_df = pd.DataFrame(list_nodes_info, columns=columns)
            nodes_df = nodes_df.sort_values(by='odds ratio', ascending=False)
            nodes_df.to_csv(folder_dir + 'OddsRatio.csv')

            print("Wrote Odds Ratios to: " + folder_dir + "OddsRatio.csv")