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

import torch
# from rdkit.Chem.rdmolfiles import MolFromSmiles
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder  # Given at start
from ogb.utils import features  # To get my features
from ogb.utils import mol  # To make the SMILES into a tensor
from ogb.utils import torch_util  # If needed, to create numpy to tensor
from ogb.graphproppred import PygGraphPropPredDataset  # To get my dataset to work with

from rdkit import Chem
import networkx as nx
from torch_geometric.data import Data

'''
Notes:

Node Embeddings will not work for the purposes of this, the program crashes early due to there being too many data points

Birch Clusterer seems to work best as it captures all embeddings
DBSCAN Clusterer removes many embeddings

Issues seem to arise in the generating graph embeddings, the method of computing the mean of all node embeddings found online does not seem to work well
I need to find and familiarize myself with a graph embedding method that works with my dataset
So far any that I have tried have not produced meaningful results
'''


# Given at start
atom_encoder = AtomEncoder(emb_dim = 100)  # Class object for node embeddings
bond_encoder = BondEncoder(emb_dim = 100)  # Class object for edge embeddings
# These automatically scale the data with an Xavier Uniform Distribution

dataset = PygGraphPropPredDataset(name='ogbg-molhiv',root='\OGB_Dataset')  # Load the dataset
print(dataset[0])

# Generate labels for graph

labels = pd.read_csv('OGB_Dataset\ogbg_molhiv\mapping\mol.csv.gz', compression='gzip').iloc[:,0]  # Get the graph labels
labels = labels.to_list()

count_ones = labels.count(1)
count_zeros = labels.count(0)

print("Number of 1s:", count_ones)  # display number of ones in the dataset
print("Number of 0s:", count_zeros)  # display number of zeros in the dataset

tooltip_s = np.array(
    labels
)  # need to make sure to feed it as a NumPy array, not a list

nplist_graph_embeddings = []
nparray_graph_embeddings = np.array([])
graph_list = []

# For all graphs in the dataset
for graph in range(len(dataset)):
    #print(graph) # To show progress
    # Get the embeddings for current SMILES
    atom_embedding = atom_encoder(dataset[graph].x)  # Get the embedding of given SMILES equation
    bond_embedding = bond_encoder(dataset[graph].edge_attr)

    num_nodes = atom_embedding.shape[0]
    embedding_dim = atom_embedding.shape[1]

    # Create PyG data object with node embeddings
    data = Data(x=atom_embedding, edge_index=dataset[graph].edge_index) 
    
    data.edge_attr = bond_embedding
    graph_embedding = torch.mean(data.x, dim=0)  # This just takes a mean of the node emeddings
    np_array = graph_embedding.detach().numpy()

    # Initialize array of graph embeddings
    if(nparray_graph_embeddings.size == 0):
        nparray_graph_embeddings = np_array

    # Add the subsequent arrays
    else:
        nplist_graph_embeddings.append(np_array)

nparray_graph_embeddings = np.stack(nplist_graph_embeddings)  # Get the graph embeddings into one object
print(nparray_graph_embeddings.shape)  # Verify this worked properly

tooltip_s = np.array(
    labels
)  # need to make sure to feed it as a NumPy array, not a list

print(len(tooltip_s))


# Initialize the mapper object
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
#projected_data = mapper.fit_transform(nparray_graph_embeddings, projection=sklearn.manifold.TSNE(), scaler=None)
projected_data = mapper.fit_transform(nplist_graph_embeddings, projection=sklearn.manifold.TSNE(), scaler=None)





'''
# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.KMeans(n_clusters=5, random_state=42), 
    cover=km.Cover(10, 0.6), 
)
# Original clusterer
# clusterer=sklearn.cluster.DBSCAN(eps=0.7, min_samples=25),  # Min samples is mid things per node, eps is the max distance between two samples to group


# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="OGB_Dataset\Visualizations\digits_ylabel_tooltips.html",
    custom_tooltips=tooltip_s,
) 

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    #graph, # original
    title="Handwritten digits Mapper",
    path_html="OGB_Dataset\Visualizations\digits_custom_tooltips.html",
    color_values=labels,  
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)
'''



# From attempting atom embeddings
'''
for node in range(len(dataset)):
    # Get the embeddings for current SMILES
    atom_embedding = atom_encoder(dataset[node].x)  # Get the embedding of given SMILES equation
    tlist_atom_embeddings.append(atom_embedding)  # Add tensor to our list of tensors

nplist_atom_embeddings.append(tensor.detach().numpy() for tensor in tlist_atom_embeddings)  # Convert tensors to numpy arrays and add to list
nplist_atom_embeddings = [tensor.detach().numpy() for tensor in tlist_atom_embeddings]  # Convert tensors to numpy arrays and add to list

nplist_atom_embeddings = np.asarray(nplist_atom_embeddings)
nplist_atom_embeddings = np.concatenate(nplist_atom_embeddings, axis=0)  # Make this into a numpy array

print(type(nplist_atom_embeddings))
print(nplist_atom_embeddings[0])

labels = []
with open('OGB_Dataset\Created_Files\Graph_Indicator.txt', "r") as f:
    for line in f:
        labels.append(f.readline())
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