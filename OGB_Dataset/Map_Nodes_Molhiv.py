"""

Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips.

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_

"""

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
atom_encoder = AtomEncoder(emb_dim = 40)  # Class object for node embeddings
# These automatically scale the data with an Xavier Uniform Distribution

dataset = PygGraphPropPredDataset(name='ogbg-molhiv',root='/OGB_Dataset')  # Load the dataset
print(dataset[0])

# Generate labels for graph

labels = pd.read_csv('C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/OGB_Dataset/ogbg_molhiv/mapping/mol.csv.gz', compression='gzip').iloc[:,0]  # Get the graph labels
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
    #clusterer=sklearn.cluster.KMeans(n_clusters=5, random_state=42, n_init=1),
    #clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=50),
    clusterer=sklearn.cluster.OPTICS(min_samples=50),
    #cover=km.Cover(10, 0.6))
    cover=km.Cover(n_cubes=5, perc_overlap=0.4))
# Creat a networkX graph for TDA mapper graph, in this graph nodes will be the clusters and the node featre would be the cluster size

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
#print("Output graph examples to html")

# Tooltips with the target y-labels for every cluster member

mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/OGB_Dataset/Visualizations/digits_ylabel_tooltips.html",
    custom_tooltips=tooltip_s,
) 

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/OGB_Dataset/Visualizations/digits_custom_tooltips.html",
    color_values=binary_labels,  
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)

# removing all the nodes without any edges (Just looking at the links) # You can change the logic based on your needs 

graph_copy = graph
# Prints out all nodes in their clusters with their node labels, what I think I will want to do is, for each node in node,
#  then give them their graph label from the txt file

for key in graph_copy["nodes"].keys():
    for value in range(len(graph_copy["nodes"][key])):
        graph_copy["nodes"][key][value] = str(binary_labels[graph_copy["nodes"][key][value]])

    
total_ones = 0
total_zeros = 0

for key in graph_copy["nodes"]:
    total_ones += graph_copy["nodes"][key].count("1")
    total_zeros += graph_copy["nodes"][key].count("0")

print("Total Ones is: ", total_ones)
print("Total Zeros is: ", total_zeros)

list_nodes_info = []

for key in graph_copy["nodes"]:
    print(key, " size is: ", len(graph_copy["nodes"][key]))
    ones_in_cluster = graph_copy["nodes"][key].count("1")
    zeros_in_cluster = graph_copy["nodes"][key].count("0")

    ones_not_in_cluster = total_ones - ones_in_cluster
    zeros_not_in_cluster = total_zeros - zeros_in_cluster

    print("Number of 1s:", ones_in_cluster)  # display number of ones in the dataset
    print("Number of 0s:", zeros_in_cluster)  # display number of zeros in the dataset

    if(zeros_in_cluster != 0):
        odds_ratio = (ones_in_cluster / zeros_in_cluster) / (ones_not_in_cluster / zeros_not_in_cluster)
    else:
        odds_ratio = 100

    print("Odds Ratio is: ", odds_ratio)

    list_nodes_info.append((key, ones_in_cluster, zeros_in_cluster, odds_ratio))


# Define column names (optional, but recommended)
columns = ["key", "number of ones", "number of zeros", "odds ratio"]

# Create DataFrame from list of tuples
nodes_df = pd.DataFrame(list_nodes_info, columns=columns)
nodes_df = nodes_df.sort_values(by='odds ratio', ascending=False)
nodes_df.to_csv('C:/Users/ronan/OneDrive/Documents/GitHub/test/Kepler_Mapper/OddsRatioInit1(2).csv', index=False)




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