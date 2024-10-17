import numpy as np
import sklearn
import kmapper as km
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import shutil
import networkx as nx
import pandas as pd 
import datetime as dt
from node2vec import Node2Vec  # Might use, uncertain as of now
from collections import defaultdict

#from kiarash_codes.rnn import 

class MyNetworkParser:
    file_path = ""
    timeseries_data_path = "../my_work/data/"
    timeseries_file_path_other = ""
    timeWindow = [7]
    # Validation duration condition
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10
    
    
    def create_time_series_sequence(self, file):
        # Read in the entire dataframe
        data = pd.read_csv(self.timeseries_data_path + file)
        data = data[['source', 'destination', 'weight', 'date']]
        data['source'] = data['source'].apply(lambda x: int(x, 16))
        data['destination'] = data['destination'].apply(lambda x: int(x, 16))

        # Values to return
        totalRnnSequenceData = list()
        totalRnnLabelData = list()

        # Parameters
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        print("Processing {}".format(file))

        # Preprocess dataset
        selectedNetwork = pd.read_csv((self.timeseries_data_path + file), sep=',', names=["source", "destination", "date", "weight"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['weight'] = selectedNetwork['weight'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        print(f"{file} -- {window_start_date} -- {data_last_date}")

        print("\n {} Days OF Data -> {} ".format(file, (data_last_date - window_start_date).days))
        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < maxDuration):
            print(file + "Is not a valid network")
            shutil.move(self.file_path + file, self.file_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['value'].max())
        min_transfer = float(selectedNetwork['value'].min())

        # Normalization function
        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1

            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[(selectedNetwork['date'] >= window_start_date) & (selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # Likely unnecessary for our uses
            '''
            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0
            '''

            using_node2vec = False
            if using_node2vec:
                node_features = self.embed_node2vec()
            else:
                node_features = self.embed_structure(selectedNetwork, selectedNetworkInGraphDataWindow, windowSize, transactionGraph)

            timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow, node_features)

            totalRnnSequenceData.append(timeWindowSequence)
            #totalRnnLabelData.append(label)  # Edit
            window_start_date = window_start_date + dt.timedelta(days=1)

        # Store data for later prediction, don't think I'll need to change
        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + '/seq_tda_ablation.txt',
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()


    def embed_node2vec(self, graph, dim, walk_len, num_walks, p, q, window):
        """
            Embed nodes using the Node2Vec class object. Implementing so that we have options later

            Args:
                graph (nx.MultiDiGraph()): The graph we are seeking to embed
                dim (int): The dimension of our embeddings
                walk_len (int): The length of each random walk
                num_walks (int): The number of random walks per node
                p (float): The likelihood of returning to the previous node
                q (float): The likelihood of visiting nodes far from starting node
                window (int): The number of neighboring nodes considered around target node during training

            Returns:
                embeddings (np.array()): An array of node embeddings for the given graph
        """
        node2vec = Node2Vec(graph, dimensions=dim, walk_length=walk_len, num_walks=num_walks, p=p, q=q)
        node2vec_model = node2vec.fit(window=window, min_count=1, batch_words=4)
            
        embeddings = np.array([node2vec_model.wv[str(node)] for node in graph.nodes()])

        return embeddings  # Need to edit to be a dataframe


    def embed_structure(self, selectedNetwork, selectedNetworkInGraphDataWindow, windowSize, transactionGraph):
        # group by for extracting node features
        outgoing_weight_sum = (selectedNetwork.groupby(by=['from'])['value'].sum())
        incoming_weight_sum = (selectedNetwork.groupby(by=['to'])['value'].sum())
        outgoing_count = (selectedNetwork.groupby(by=['from'])['value'].count())
        incoming_count = (selectedNetwork.groupby(by=['to'])['value'].count())

        # Node Features Dictionary for TDA mapper usage
        node_features = pd.DataFrame()

        # Populate graph with edges
        for item in selectedNetworkInGraphDataWindow.to_dict(orient="records"):
            from_node_features = {}
            to_node_features = {}
            # calculating node features for each edge
            # feature 1 -> sum of outgoing edge weights
            from_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['from']]

            try:
                to_node_features["outgoing_edge_weight_sum"] = outgoing_weight_sum[item['to']]
            except Exception as e:
                to_node_features["outgoing_edge_weight_sum"] = 0

            # feature 2 -> sum of incoming edge weights
            to_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['to']]
            try:
                from_node_features["incoming_edge_weight_sum"] = incoming_weight_sum[item['from']]
            except Exception as e:
                from_node_features["incoming_edge_weight_sum"] = 0
            # feature 3 -> number of outgoing edges
            from_node_features["outgoing_edge_count"] = outgoing_count[item['from']]
            try:
                to_node_features["outgoing_edge_count"] = outgoing_count[item['to']]
            except Exception as e:
                to_node_features["outgoing_edge_count"] = 0

            # feature 4 -> number of incoming edges
            to_node_features["incoming_edge_count"] = incoming_count[item['to']]
            try:
                from_node_features["incoming_edge_count"] = incoming_count[item['from']]
            except Exception as e:
                from_node_features["incoming_edge_count"] = 0

            # add temporal vector to all nodes, populated with -1

            from_node_features_with_daily_temporal_vector = dict(from_node_features)
            from_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
            from_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

            to_node_features_with_daily_temporal_vector = dict(to_node_features)
            to_node_features_with_daily_temporal_vector["dailyClusterID"] = [-1] * windowSize
            to_node_features_with_daily_temporal_vector["dailyClusterSize"] = [-1] * windowSize

            # Temporal version
            transactionGraph.add_nodes_from(
                [(item["from"], from_node_features_with_daily_temporal_vector)])
            transactionGraph.add_nodes_from([(item["to"], to_node_features_with_daily_temporal_vector)])
            transactionGraph.add_edge(item["from"], item["to"], value=item["value"])

            new_row = pd.DataFrame(({**{"nodeID": item["from"]}, **from_node_features}), index=[0])
            node_features = pd.concat([node_features, new_row], ignore_index=True)

            new_row = pd.DataFrame(({**{"nodeID": item["to"]}, **to_node_features}), index=[0])
            node_features = pd.concat([node_features, new_row], ignore_index=True)

            node_features = node_features.drop_duplicates(subset=['nodeID'])

        return node_features  # Return the dataframe of node features
    
    
    # Needs editing, ask Kiarash if he wants me to remove the nodes without edges
    def extract_tda_features(self, tda_graph):
        # removing all the nodes without any edges (Just looking at the links)
        tdaGraph = nx.Graph()
        for key, value in tda_graph['links'].items():
            tdaGraph.add_nodes_from([(key, {"cluster_size": len(tda_graph["nodes"][key])})])
            for to_add in value:
                tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(tda_graph["nodes"][to_add])})])
                tdaGraph.add_edge(key, to_add)

        features = {}  # Return value, is dict for readability

        # Used for collecting features from tda graph
        nx_graph = self.tda_to_networkx(tda_graph)  
        links = tda_graph['links']
        nodes = list(tda_graph['nodes'].keys()) 

        # Get features
        try:
            features['num_nodes'] = nx_graph.number_of_nodes()
            features['num_edges'] = nx_graph.number_of_edges()
            features['density'] = nx.density(nx_graph)

            # For cluster information
            cluster_info = []
            for cluster_id, node_list in nodes.items():
                cluster_size = len(node_list)
                cluster_info.append(cluster_size)

            cluster_info = np.array(cluster_info)
            features['max_cluster_size'] = np.max(cluster_info)
            features['avg_cluster_size'] = np.average(cluster_info)

            # Connected component information
            connected_components = list(nx.connected_components(nx_graph))
            largest_connected_component = max(connected_components, key=len)
            features['num_connected_components'] = len(connected_components)
            features['largest_connected_component'] = len(largest_connected_component)

            # average edge weight
            edge_weights = defaultdict(dict)
            for source_node, target_nodes in tda_graph['links'].items():
                for target_node in target_nodes:
                    common_indexes = len(
                        set(tda_graph['nodes'][source_node]) & set(tda_graph['nodes'][target_node]))
                    edge_weights[source_node][target_node] = common_indexes
            total_edge_weights = sum(
                weight for target_weights in edge_weights.values() for weight in target_weights.values())
            total_edges = sum(len(target_weights) for target_weights in edge_weights.values())
            features['average_edge_weight'] = total_edge_weights / total_edges


        except Exception as e:
            features['num_nodes'] = 0
            features['num_edges'] = 0
            features['density'] = 0
            features['max_cluster_size'] = 0
            features['avg_cluster_size'] = 0
            features['num_connected_components'] = 0
            features['largest_connected_component'] = 0
            features['average_edge_weight'] = 0

        return features.values()
    
    
    # Not filling out for now, depends if we want to concatenate original graph information like they do in graphpulse
    def extract_graph_features(self):
        pass
    
    
    # Ignoring for now, can do at a later date if necessary
    def extract_node_features(self, graph):
        pass
    

    def tda_to_networkx(graph):
        nx_graph = nx.Graph()
        
        # Add nodes and cluster size attribute
        for node, cluster in graph['nodes'].items():
            nx_graph.add_node(node, size=len(cluster))

        # Add edges to the graph
        for edge in graph['links']:
            nx_graph.add_edge(edge[0], edge[1])
        
        return nx_graph
    

    def tda_process(self, mapper, lens, Xfilt, per_overlap, n_cubes, cls, node_ids):
        graph = mapper.map(
            lens,
            Xfilt,
            clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=42),
            cover=km.Cover(n_cubes=n_cubes, perc_overlap=per_overlap),  
        )

        # If you want a visualization, uncomment this
        '''
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
        '''

        return self.extract_tda_features(graph)
    
    
    def process_TDA_extracted_rnn_sequence(self, timeFrameData, nodeFeatures):
        # break the data to daily graphs
        timeWindowSequence = list()
        data_first_date = timeFrameData['date'].min()
        data_last_date = timeFrameData['date'].max()
        numberOfDays = (data_last_date - data_first_date).days
        start_date = data_first_date
        # initiate the graph
        processingDay = 0

        # Loop over each day in the snapshot
        while (processingDay <= numberOfDays):
            # print("Processing TDA RNN sequential Extraction day {}".format(processingDay))
            daily_end_date = start_date + dt.timedelta(days=1)
            selectedDailyNetwork = timeFrameData[
                (timeFrameData['date'] >= start_date) & (timeFrameData['date'] < daily_end_date)]

            daily_node_features = pd.DataFrame()

            for item in selectedDailyNetwork.to_dict(orient="records"):
                new_row = pd.DataFrame(({**{"nodeID": item["from"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                new_row = pd.DataFrame(({**{"nodeID": item["to"]},
                                         **nodeFeatures[nodeFeatures["nodeID"] == item["to"]].drop("nodeID",
                                                                                                   axis=1).to_dict(
                                             orient='records')[0]}),
                                       index=[0])
                daily_node_features = pd.concat([daily_node_features, new_row], ignore_index=True)

                daily_node_features = daily_node_features.drop_duplicates(subset=['nodeID'])

            # creat the TDA for each day
            try:
                Xfilt = daily_node_features
                Xfilt = Xfilt.drop(columns=['nodeID'])
                mapper = km.KeplerMapper()
                scaler = MinMaxScaler(feature_range=(0, 1))

                Xfilt = scaler.fit_transform(Xfilt)
                lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())

                # List to store the result objects
                results = []

                per_overlap = 0.05
                n_cubes = 10
                cls = 5
                for albation_index in [1]:
                    # per_overlap = round(per_overlap_indx * 0.05, 2)
                    result = self.TDA_process(mapper, lens, Xfilt, per_overlap, n_cubes, cls)
                    results.append(result)

                # Retrieve the results as they become available
                for result in results:
                    dailyFeatures = result.get()
                    timeWindowSequence.append(dailyFeatures)


            except Exception as e:
                print(str(e))

            start_date = start_date + dt.timedelta(days=1)
            processingDay += 1

        # the graph has been repopulated with daily temporal features
        merged_dict = self.merge_dicts(timeWindowSequence)
        return merged_dict


    # Utility function, used to build the sequence/label pair
    def merge_dicts(self, list_of_dicts):
        merged_dict = {}
        for dictionary in list_of_dicts:
            for key, value in dictionary.items():
                if key in merged_dict:
                    merged_dict[key].append(value)
                else:
                    merged_dict[key] = [value]
        return merged_dict




'''
Generate Snapshots
Get a dataframe of either strucutral encodings or node2vec embeddings where I can query based on the nodeid in question to build a daily graph
For Each snapshot, generate a TDA graph for each daily graph based on their roles in the graph for the entire snapshot
Generate graph-level features for the TDA graph (These will act as the labels for our RNN)
Still need to figure out what our input features are for the RNN (honestly the labels could be both)
Run data through the RNN
'''