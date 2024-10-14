import numpy as np
import networkx as nx
import pandas as pd 
from node2vec import Node2Vec
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.decomposition import TruncatedSVD

prepend = 'my_work/Outputs/SVD/'

def createGraph(curr_data):
    curr_graph = nx.DiGraph()
    
    # Generate embeddings
    for idx, row in curr_data.iterrows():
        curr_graph.add_edge(row['source'], row['destination'], value=row['weight'])
        
    return curr_graph


def performSVD(graph, num_values, timestamp):
    A = nx.adjacency_matrix(graph).todense()  # Get adjacency matrix

    # U, Sigma, VT = np.linalg.svd(A)
    svd = TruncatedSVD(n_components=num_values)  # Specify the number of components to keep
    
    reduced_matrix = svd.fit_transform(A)
    mat_df = pd.DataFrame(reduced_matrix, columns=[f'Component_{i+1}' for i in range(num_values)])
    mat_df.to_csv(prepend + str(timestamp) + '.csv', index=False)



# Read data
data = pd.read_csv('my_work/ARC_edgelist.csv')
data = data[['source', 'destination', 'weight', 'date']]
data['source'] = data['source'].apply(lambda x: int(x, 16))
data['destination'] = data['destination'].apply(lambda x: int(x, 16))

timestamps = np.unique(data['date'].values)
#timestamps = timestamps[70:]

for timestamp in timestamps:
    graph = createGraph(data[data['date'] == timestamp])
    num_values = (graph.number_of_nodes() * 3) // 4
    performSVD(graph, num_values, timestamp)