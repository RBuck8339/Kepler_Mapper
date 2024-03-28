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

dir='C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/'


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
atom_encoder = AtomEncoder(emb_dim = 60)  # Class object for node embeddings
# These automatically scale the data with an Xavier Uniform Distribution

dataset = PygGraphPropPredDataset(name='ogbg-molhiv',root='/OGB_Dataset')  # Load the dataset
print(dataset[0])
print(len(dataset))
# Generate labels for graph

labels = pd.read_csv(dir + 'OGB_Dataset/ogbg_molhiv/mapping/mol.csv.gz', compression='gzip').iloc[:,0]  # Get the graph labels
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

# Fit and transform data
projected_data = mapper.fit_transform(nplist_atom_embeddings, projection=sklearn.manifold.TSNE(), scaler=None)

graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.KMeans(n_clusters=10, random_state=42, n_init=1),
    #clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=50),
    #clusterer=sklearn.cluster.OPTICS(min_samples=50),
    cover=km.Cover(10, 0.6))
    #cover=km.Cover(n_cubes=5, perc_overlap=0.4))
# Creat a networkX graph for TDA mapper graph, in this graph nodes will be the clusters and the node featre would be the cluster size

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
#print("Output graph examples to html")

# Tooltips with the target y-labels for every cluster member

mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html=dir + "Outputs\Visualizations\digits_ylabel_tooltips.htmll",
    custom_tooltips=tooltip_s,
) 

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html=dir + "Outputs\Visualizations\digits_custom_tooltips.html",
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
num_members_df.to_csv('Outputs/NumMembers.csv', index=False)

print("Wrote dataframe containing number of members in each cluster to: Outputs/NumMembers.csv")

'''
count_nodes_keys = graph["nodes"].keys()
members_df = pd.DataFrame(graph_memberships, columns=count_nodes_keys)
members_df.to_csv()
'''

count_graph_nodes_dict = dict.fromkeys(graph["nodes"].keys())


list_sizes = ["0-5 nodes", "6-15 nodes", "16-30 nodes", "31-60 nodes", "61-100 nodes", "101-250 nodes", "251-500 nodes", "501-1000 nodes", "1001-2000 nodes", 
              "2001-4000 nodes", "4001-6000 nodes", "6001-10000 nodes", "10001-15000 nodes", "15001-20000 nodes", "20001-30000 nodes", 
              "30001-45000 nodes", "45001-60000 nodes", "60001-80000 nodes", "80001-100000 nodes", "100001-150000 nodes", "150001+ nodes"]

list_vals = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

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

count_graph_nodes_df.to_csv('Outputs\ClusterSizes.csv', index=False)

print("Wrote dataframe containing cluster sizes to Outputs\ClusterSizes.csv")
  
graph_copy = graph

graph_indicators = []
with open('OGB_Dataset\Created_Files\Graph_Indicator.txt', "r") as f:
    for line in f:
        graph_indicators.append(int(line))

# Get all column names
columns = ["graph number"]
columns = columns + list(graph["nodes"].keys())

# Modify the graph_copy to have graph indicators so we can count where the nodes appear
for key, values in graph_copy["nodes"].items():
    for i in range(len(values)):
        values[i] = graph_indicators[values[i] - 1]

num_nodes_cluster_df = pd.DataFrame(columns=columns)

for graph in range(len(dataset)):
    new_row = []
    new_row.append(graph)
    for key in graph_copy["nodes"].keys():
        new_row.append((graph_copy["nodes"][key]).count(graph))
    num_nodes_cluster_df.loc[len(num_nodes_cluster_df)] = new_row
    
num_nodes_cluster_df.to_csv('Outputs\GraphNodeDistributions.csv', index=False)  

print("Worte dataframe containing node distributions to Outputs\GraphNodeDistributions.csv")

new_graph = graph  # Reset graph_copy

# Prints out all nodes in their clusters with their node labels, what I think I will want to do is, for each node in node,
#  then give them their graph label from the txt file

for key in list(new_graph["nodes"].keys()):
    for value in range(len(list(new_graph["nodes"][key]))):
        new_graph["nodes"][key][value] = str(binary_labels[new_graph["nodes"][key][value]])

    
total_ones = 0
total_zeros = 0

for key in new_graph["nodes"]:
    total_ones += new_graph["nodes"][key].count("1")
    total_zeros += new_graph["nodes"][key].count("0")


list_nodes_info = []

for key in new_graph["nodes"]:
    #print(key, " size is: ", len(graph_copy["nodes"][key]))
    ones_in_cluster = new_graph["nodes"][key].count("1")
    zeros_in_cluster = new_graph["nodes"][key].count("0")

    ones_not_in_cluster = total_ones - ones_in_cluster
    zeros_not_in_cluster = total_zeros - zeros_in_cluster

    if(zeros_in_cluster != 0):
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
nodes_df.to_csv(dir + 'Outputs\OddsRatio.csv', index=False)

print("Wrote Odds Ratios to: Outputs\OddsRatio.csv")


'''
#From Kiarash

tdaGraph = nx.Graph()

for key, value in graph['links'].items():
    tdaGraph.add_nodes_from([(key, {"cluster_size": len(graph["nodes"][key])})])
    for to_add in value:
        tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(graph["nodes"][to_add])})])
        tdaGraph.add_edge(key, to_add)


'''


'''
# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.KMeans(n_clusters=5, random_state=42, n_init=1), 
    cover=km.Cover(10, 0.60), 
)
# Original clusterer
# clusterer=sklearn.cluster.DBSCAN(eps=0.7, min_samples=25),  # Min samples is mid things per node, eps is the max distance between two samples to group

'''




# Keep for reference


'''
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')  # Load the dataset
smiles_strings = pd.read_csv('OGB_Dataset\ogbg_molhiv\mapping\mol.csv.gz', compression='gzip').iloc[:,1]  # Get the smiles strings from the csv
smiles_string = mol.smiles2graph(smiles_strings[0])  # Extract the first smile string from the series
print(dataset[0])
print(type(dataset[0].edge_attr))
print(type(smiles_string["edge_feat"]))
smiles_node = torch_util.replace_numpy_with_torchtensor(smiles_string["node_feat"])  # Makes the smile string into a tensor
smiles_edge = torch_util.replace_numpy_with_torchtensor(smiles_string["edge_feat"])  # Makes the smile string into a tensor

# These are equivalent
#print(atom_encoder(smiles_node))  # I get a single long tensor out of this
#print(atom_encoder(dataset[0].x))
#print(bond_encoder(smiles_edge))  # I get a single long tensor out of this
#print(bond_encoder(dataset[0].edge_attr))
'''