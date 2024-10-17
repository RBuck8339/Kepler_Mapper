import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


path = "my_work/CoarseningOutputs/"

def createGraph(curr_data):
    curr_graph = nx.DiGraph()
    
    # Generate embeddings
    for idx, row in curr_data.iterrows():
        curr_graph.add_edge(row['from'], row['to'], value=row['value'])
        
    return curr_graph
        

def groupNodes(graph):
    coarsened_graph = nx.DiGraph()
    coarsened_nodes = {}

    # Step 1: Group nodes based on their successors
    for node in graph.nodes:
        # Get the successors of the current node
        successors = tuple(sorted(graph.successors(node)))

        # Check if there are successors to consider
        if successors:
            if successors in coarsened_nodes:
                coarsened_nodes[successors].append(node)
            else:
                coarsened_nodes[successors] = [node]

    # Step 2: Create coarsened nodes only if they can be merged
    merging_occurred = False

    for merged_successors, original_nodes in coarsened_nodes.items():
        if len(original_nodes) > 1:
            # Create a new node to represent the merged nodes
            new_node = "_".join(map(str, original_nodes))
            coarsened_graph.add_node(new_node)

            # Add edges from the new node to the common successors
            for successor in merged_successors:
                coarsened_graph.add_edge(new_node, successor)

            merging_occurred = True
        else:
            # Keep the node if it doesn't need to be merged
            original_node = original_nodes[0]
            coarsened_graph.add_node(original_node)
            for successor in merged_successors:
                coarsened_graph.add_edge(original_node, successor)

    if not merging_occurred:
        return graph

    return coarsened_graph


def checkOutputs(graph, type):
    print(f'There are currently {graph.number_of_nodes()} nodes in this graph')
    
    # Might need to edit depending on how outputs look
    pos = nx.spring_layout(graph)  # You can choose different layouts (e.g., circular, shell, etc.)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, font_color='black')
    plt.title("Ethereum Actual Map for: " + str(timestamp))
    plt.savefig(path + "ethereum_map_"  + str(timestamp) + type +".png", format="PNG")


def performSVD(graph):
    A = nx.adjacency_matrix(graph).todense()

    # Perform Singular Value Decomposition (SVD)
    U, Sigma, VT = np.linalg.svd(A)

    # U contains the left singular vectors
    # Sigma contains the singular values
    # VT contains the right singular vectors
    print("U (Left Singular Vectors):\n", U)
    print("Sigma (Singular Values):\n", Sigma)
    print("VT (Right Singular Vectors):\n", VT)


data = pd.read_csv('my_work/EthereumDataset.csv')


timestamps = data['timestamp'].values
timestamps = np.unique(timestamps)

for timestamp in timestamps:
    graph = nx.DiGraph()
    curr_data = data[data['timestamp'] == timestamp]

    graph = createGraph(curr_data)
    
    # Since no edits can be done with a graph that small
    if graph.number_of_nodes() <= 2:
        continue
    
    print(f'Timestamp is currently: {timestamp}')
    checkOutputs(graph, type='_actual')
    
    plt.close()
    modified_graph = groupNodes(graph)
    
    checkOutputs(modified_graph, type='_modified')
    plt.close()
    
    performSVD(modified_graph)