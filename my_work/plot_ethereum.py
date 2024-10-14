import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import kmapper as km

# My imports
import networkx as nx
import pandas as pd 
from node2vec import Node2Vec
from networkx.drawing.nx_pydot import graphviz_layout


# Read data
data = pd.read_csv('my_work/ARC_edgelist.csv')
data = data[['source', 'destination', 'weight', 'date']]
data['source'] = data['source'].apply(lambda x: int(x, 16))
data['destination'] = data['destination'].apply(lambda x: int(x, 16))


timestamps = np.unique(data['date'].values)
#timestamps = timestamps[70:]

for timestamp in timestamps:
    curr_data = data[data['date'] == timestamp]
    
    curr_graph = nx.DiGraph()
    
    # Generate embeddings
    for idx, row in curr_data.iterrows():
        curr_graph.add_edge(row['source'], row['destination'], value=row['weight'])
    
    if(curr_graph.number_of_nodes() <= 30):
        print(f"Skipping {str(timestamp)}")
        continue
    
    print(f"Num nodes: {curr_graph.number_of_nodes()}")
    
    '''
    Node2Vec parameter Guide:
        p (return parameter): Controls the likelihood of returning to the previous node. A higher p makes the walk more depth-first (exploring fewer, more connected nodes).
        q (in-out parameter): Controls the likelihood of visiting nodes far away from the starting node. A lower q makes the walk more breadth-first (exploring many nodes, but not in-depth).
        walk_length: Length of each random walk starting from a node.
        num_walks: Number of random walks per node.
        dimensions: Number of dimensions for the node embeddings.
        window_size: Context size for optimization (used in Skip-gram model).
        workers: Number of CPU cores to use for parallelizing walks.
        window: the number of neighboring nodes considered around the target node during training (helps learn which nodes appear near each other)
        min_count: the number of occurrence of a node in random walks for it to be considered during training
        batch_words: number of nodes processed per batch during training
    '''
    

    if curr_graph.number_of_nodes() < 80:  # Check if there are enough embeddings to proceed
        node2vec = Node2Vec(curr_graph, dimensions=32, walk_length=10, num_walks=50, p=1, q=0.8)
        node2vec_model = node2vec.fit(window=3, min_count=1, batch_words=4)
            
        embeddings = np.array([node2vec_model.wv[str(node)] for node in curr_graph.nodes()])
        node_ids = list(curr_graph.nodes())

        # Plug into KeplerMapper
        mapper = km.KeplerMapper(verbose=2)

        # Fit and transform embeddings data using t-SNE for better visualization
        projected_data = mapper.fit_transform(embeddings, projection=sklearn.manifold.TSNE())
        
        graph = mapper.map(
            projected_data,
            embeddings,  # Maybe try using embeddings here as well
            clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=1),  
            cover=km.Cover(n_cubes=10, perc_overlap=0.15),  
        )
    else:
        node2vec = Node2Vec(curr_graph, dimensions=64, walk_length=25, num_walks=25, p=1.5, q=1.5)
        node2vec_model = node2vec.fit(window=5, min_count=1, batch_words=4)
            
        embeddings = np.array([node2vec_model.wv[str(node)] for node in curr_graph.nodes()])
        node_ids = list(curr_graph.nodes())

        # Plug into KeplerMapper
        mapper = km.KeplerMapper(verbose=2)

        # Fit and transform embeddings data using t-SNE for better visualization
        
        projected_data = mapper.fit_transform(embeddings, projection=sklearn.manifold.TSNE())
        graph = mapper.map(
            projected_data,
            embeddings,  # Maybe try using embeddings here as well
            clusterer=sklearn.cluster.DBSCAN(eps=0.5, min_samples=1),  
            cover=km.Cover(n_cubes=10, perc_overlap=0.1),  
        )

    prepend = "my_work/Outputs/DBScan/"  # For output path

    # Create the visualizations
    print("Output graph examples to html")
    custom_tooltips = [str(node_id) for node_id in node_ids]  # Use node IDs as tooltips
    custom_tooltips = np.array(custom_tooltips)  # Necessary or else mapper will crash
    
    mapper.visualize(
        graph,
        title="Ethereum Nodes Mapper",
        path_html=prepend + "ethereum_map_" + str(timestamp) + ".html",
        color_values=node_ids,  # Optional, color based on node IDs
        color_function_name="Node IDs",
        custom_tooltips=custom_tooltips,  # Must be a numpy array
    )
    
    # Don't need on this big of a graph
    '''
    # Make a matplotlib png for comparison
    pos = nx.spring_layout(curr_graph)  # You can choose different layouts (e.g., circular, shell, etc.)
    nx.draw(curr_graph, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=1, font_color='black')
    plt.title("Ethereum Actual Map for: " + str(timestamp))
    plt.savefig(prepend + "ethereum_map_" + str(timestamp) +".png", format="PNG")
    '''
    
    # Create adjacency matrix
    links = graph['links']
    nodes = list(graph['nodes'].keys()) 

    # Initialize adjacency matrix
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}  # Map node to index

    for node, neighbors in links.items():
        for neighbor in neighbors:
            # get node ids
            i, j = node_to_idx[node], node_to_idx[neighbor]
            # Set the value in the adjacency matrix to 1 for each edge
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  

    # Save as csv
    adj_df = pd.DataFrame(adj_matrix, index=nodes, columns=nodes)
    adj_df.to_csv(prepend + 'ethereum_map_' + str(timestamp) + '_adjacency.csv', index=True)
    
# Reference code https://github.com/cakcora/MultiverseJ/blob/master/python/MultiverseBinaryCode.py 