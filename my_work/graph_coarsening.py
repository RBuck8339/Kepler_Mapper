import networkx as nx
import matplotlib as plt
import pandas as pd
import numpy as np


def createGraph(data):
    pass


def groupNodes(graph):
    pass


def checkOutputs(graph):
    pass 


data = pd.read_csv('EthereumDataset.csv')


timestamps = data['timestamp'].values
timestamps = np.unique(timestamps)

for timestamp in timestamps:
    curr_data = data[data['timestamp'] == timestamp]
    
    curr_graph = nx.DiGraph()
    
    # Generate embeddings
    for idx, row in curr_data.iterrows():
        curr_graph.add_edge(row['from'], row['to'], value=row['value'])
    
    if curr_graph.number_of_nodes() <= 2:
        continue

    graph = createGraph(data)
    graph = groupNodes(graph)
    checkOutputs(graph)