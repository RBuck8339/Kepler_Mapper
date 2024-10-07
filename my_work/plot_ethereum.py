# Some imports I stole

import io
import sys
import base64

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import kmapper as km
from pathlib import Path

# My imports
import networkx as nx
import pandas as pd 
from node2vec import Node2Vec
from networkx.drawing.nx_pydot import graphviz_layout


# Read data
data = pd.read_csv('my_work/EthereumDataset.csv')
data = data[['from', 'to', 'value', 'timestamp']]
data = data.sort_values(by = 'timestamp')


# Loop over by timestamps
timestamps = data['timestamp'].values
timestamps = np.unique(timestamps)

for timestamp in timestamps:
    curr_data = data[data['timestamp'] == timestamp]
    
    curr_graph = nx.DiGraph()
    
    # Generate embeddings
    for idx, row in curr_data.iterrows():
        curr_graph.add_edge(row['from'], row['to'], value=row['value'])
    
    print(f"Num nodes: {curr_graph.number_of_nodes()}")
    
    node2vec = Node2Vec(curr_graph, dimensions=8, walk_length=3, num_walks=10, p=0.5)
    node2vec_model = node2vec.fit(window=3, min_count=1, batch_words=4)
        
    embeddings = np.array([node2vec_model.wv[str(node)] for node in curr_graph.nodes()])
    node_ids = list(curr_graph.nodes())

    # Plug into KeplerMapper
    mapper = km.KeplerMapper(verbose=2)

    # Fit and transform embeddings data using t-SNE for better visualization
    projected_data = mapper.fit_transform(embeddings, projection=sklearn.manifold.TSNE())

    if len(embeddings) > 1:  # Check if there are enough embeddings to proceed
        graph = mapper.map(
            projected_data,
            clusterer=sklearn.cluster.DBSCAN(eps=0.5, min_samples=1),  # Consider reducing min_samples
            cover=km.Cover(n_cubes=3, perc_overlap=0.4),  # Reduced n_cubes
        )
    else:
        print(f"Not enough data points for timestamp {timestamp}. Skipping...")
        continue  # Skip to the next iteration if not enough data

    # Create the visualizations
    print("Output graph examples to html")

    # Set the node IDs as tooltips
    custom_tooltips = [str(node_id) for node_id in node_ids]  # Use node IDs as tooltips  # Gotta fix

    # Check for output directory
    prepend = "my_work/Outputs/"

    # Visualize with node IDs as custom tooltips and node labels as colors (optional)
    mapper.visualize(
        graph,
        title="Ethereum Nodes Mapper",
        path_html=prepend + "ethereum_map_" + str(timestamp) + ".html",
        color_values=node_ids,  # Optional, color based on node IDs
        color_function_name="Node IDs",
        #custom_tooltips=custom_tooltips,  # Tooltip with node IDs
    )
    
    # Make a matplotlib png for comparison
    
    pos = nx.spring_layout(curr_graph)  # You can choose different layouts (e.g., circular, shell, etc.)
    nx.draw(curr_graph, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_color='black')

    plt.title("Ethereum Actual Map for: " + str(timestamp))

    # Save the graph as a PNG file
    plt.savefig("ethereum_map_" + str(timestamp) +".png", format="PNG")


'''
Gameplan

Read data into a networkx graph

Find node in_degree and out_degree
Create the node features
Could calculate the betweenness_centrality for the nodes

OR

Use node2vec

Have the node labels be the node ID's

Make the mapper
Project the node features using mapper

Edits:
    - Could make different graphs for each timestamp 
    - 

'''