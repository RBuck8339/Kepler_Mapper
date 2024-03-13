"""

Digits Dataset
================

This digits example shows two ways of customizing the tooltips options in the HTML visualization. It generates the visualization with tooltips set as the y-label, or number of the image. The second generated result uses the actual image in the tooltips.

`Visualization with y-label tooltip <../../_static/digits_ylabel_tooltips.html>`_

`Visualization with custom tooltips <../../_static/digits_custom_tooltips.html>`_

"""

# sphinx_gallery_thumbnail_path = '../examples/digits/digits-tsne-custom-tooltip-mnist.png'

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


# Given at start
atom_encoder = AtomEncoder(emb_dim = 100)  # Class object
bond_encoder = BondEncoder(emb_dim = 100)  # Class object
# These automatically scale the data with an Xavier Uniform Distribution

dataset = PygGraphPropPredDataset(name='ogbg-molhiv',root='\OGB_Dataset')  # Load the dataset

# Get jsons for graph2vec
'''
print(dataset[0])
print(dataset[0].edge_attr)

mydict = dataset[0].to_dict()
print(mydict["edge_attr"])

for graph in range(len(dataset)):
    filename = f"{'GraphEmbeddings'}/{graph}.json"
    mydict = dataset[graph].to_dict()

    for key, value in mydict.items():
        if isinstance(value, torch.Tensor):
            mydict[key] = value.detach().numpy()

    for key, value in mydict.items():
        if isinstance(value, np.ndarray):
            mydict[key] = value.tolist()

    mydict.pop("num_nodes")
    mydict.pop("y")
    mydict.pop("edge_index")
    mydict["edges"] = mydict.pop("edge_attr")
    mydict["features"] = mydict.pop("x")

    with open(filename, 'w') as outfile:
        json.dump(mydict, outfile)
'''



# Creates my graph indicators for each node
'''
graph_indicator_list = []

for graphs in range(len(dataset)):
    print(graphs)
    for nodes in range(dataset[graphs].num_nodes):
        graph_indicator_list.append(graphs + 1)

graph_indicator_df = pd.DataFrame(graph_indicator_list)
graph_indicator_df.to_csv('Graph_Indicator.txt', sep='\t', index=False)
'''


# Read in my labels
labels_df = pd.read_csv('OGB_Dataset/ogbg_molhiv/raw/graph-label.csv.gz', compression='gzip') # Make pandas dataframe
labels_list = labels_df['0'].to_list()  # Make into a list of labels

graph_indicators_list = []


'''
smiles_as_graph = []

smiles_strings = pd.read_csv('OGB_Dataset\ogbg_molhiv\mapping\mol.csv.gz', compression='gzip').iloc[:,1]  # Get the smiles strings from the csv

for string in range(len(smiles_strings)):
    smiles_as_graph.append(mol.smiles2graph(smiles_strings[string]))
'''


'''
# These store equivalent information, accessed in different ways
smiles_string = mol.smiles2graph(smiles_strings[0])  # Extracts the first smile string from the series
print(dataset[0])
print(smiles_string)
'''

# These are my embeddings
#print(atom_encoder(dataset[0].x))  # Atoms
#print(bond_encoder(dataset[0].edge_attr))  # Bonds

# For utility
#print(len(atom_encoder(dataset[0].x)))
#print(len(bond_encoder(dataset[0].edge_attr)))

tlist_atom_embeddings = []  # A list of atom embeddings expressed as pytorch tensors
nplist_atom_embeddings = []  # A list of atom embeddings expressed as numpy arrays

#list_bond_embeddings = []
#list_bond_embeddings = np.array(list_bond_embeddings)
print("Stop")
# For all nodes in the dataset
for node in range(len(dataset)):

    # Get the embeddings for current SMILES
    atom_embedding = atom_encoder(dataset[node].x)  # Get the embedding of given SMILES equation
    tlist_atom_embeddings.append(atom_embedding)  # Add tensor to our list of tensors

nplist_atom_embeddings = [tensor.detach().numpy() for tensor in tlist_atom_embeddings]  # Convert tensors to numpy arrays and add to list

nplist_atom_embeddings = np.concatenate(nplist_atom_embeddings, axis=0)  # Make this into a numpy array



# Initialize the mapper object
mapper = km.KeplerMapper(verbose=2)

# Starting here needs to be edited

'''

# Not sure what I could do with this
tooltip_s = np.array(
    tooltip_s
)  # need to make sure to feed it as a NumPy array, not a list
'''

# Fit and transform data
projected_data = mapper.fit_transform(nplist_atom_embeddings, projection=sklearn.manifold.TSNE(), scaler=None)

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),  # Min samples is mid things per node, eps is the max distance between two samples to group
    cover=km.Cover(35, 0.4), # Second paratam should be larger than max distance from DBSCAN, first is number of hypercubes, which i find from projection 
)


# I'm pretty sure that everything after here is good

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    #graph, # original
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    color_values=graph_indicators_list,  # This is in the proper data structure
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    custom_tooltips=graph_indicators_list,
) 

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



# Original Code
'''
# Load digits data
data, labels = datasets.load_digits().data, datasets.load_digits().target

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(data).astype(np.uint8)

# Create images for a custom tooltip array
tooltip_s = []
for image_data in data:
    with io.BytesIO() as output:
        img = Image.fromarray(image_data.reshape((8, 8)), "L")
        img.save(output, "PNG")
        contents = output.getvalue()
        img_encoded = base64.b64encode(contents)
        img_tag = """<img src="data:image/png;base64,{}">""".format(
            img_encoded.decode("utf-8")
        )
        tooltip_s.append(img_tag)

tooltip_s = np.array(
    tooltip_s
)  # need to make sure to feed it as a NumPy array, not a list

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")
# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    color_values=labels,
    color_function_name="labels",
    custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()

'''
'''
# Modified Code to get really big map

df = pd.read_csv("examples/data/embeddings.csv")
columns = [c for c in df.columns]
X = np.array(df[columns].fillna(0))  # quick and dirty imputation
# y = np.array(df["diagnosis"])

# Raw data is (0, 16), so scale to 8 bits (pillow can't handle 4-bit greyscale PNG depth)
scaler = MinMaxScaler(feature_range=(0, 255))
data = scaler.fit_transform(X).astype(np.float32)

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper(verbose=2)

# Fit and transform data
projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())

# Create the graph (we cluster on the projected data and suffer projection loss)
graph = mapper.map(
    projected_data,
    clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),
    cover=km.Cover(35, 0.4),
)

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)
print("Output graph examples to html")

# Tooltips with image data for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_custom_tooltips.html",
    # color_values=labels,
    color_function_name="labels",
    # custom_tooltips=tooltip_s,
)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(
    graph,
    title="Handwritten digits Mapper",
    path_html="examples\output\digits_ylabel_tooltips.html",
    # custom_tooltips=labels,
) 

# Matplotlib examples
km.draw_matplotlib(graph, layout="spring")
plt.show()
'''
