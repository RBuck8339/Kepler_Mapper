import numpy as np
import sklearn
import kmapper as km
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import shutil
# My imports
import networkx as nx
import pandas as pd 
from node2vec import Node2Vec  # Might use, uncertain as of now
#from kiarash_codes.rnn import 

class MyNetworkParser:
    file_path = ""
    timeseries_file_path = ""
    timeseries_file_path_other = ""
    timeWindow = [7]
    # Validation duration condition
    networkValidationDuration = 20
    finalDataDuration = 5
    labelTreshholdPercentage = 10
    
    data = pd.DataFrame()

    def __init__(self):
        self.data = pd.read_csv('my_work/data/ARC_edgelist.csv')
        self.data = self.data[['source', 'destination', 'weight', 'date']]
        self.data['source'] = self.data['source'].apply(lambda x: int(x, 16))
        self.data['destination'] = self.data['destination'].apply(lambda x: int(x, 16))
    
    
    def create_time_series_sequence(self, file):
        totalRnnSequenceData = list()
        totalRnnLabelData = list()
        print("Processing {}".format(file))
        windowSize = 7  # Day
        gap = 3
        lableWindowSize = 7  # Day
        maxDuration = 180  # Day
        indx = 0
        maxIndx = 2

        selectedNetwork = pd.read_csv((self.timeseries_file_path + file), sep=' ',
                                      names=["from", "to", "date", "value"])
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
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

        selectedNetwork['value'] = selectedNetwork['value'].apply(
            lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (windowSize + gap + lableWindowSize):
            print("\nRemaining Process  {} ".format(

                (data_last_date - window_start_date).days / (windowSize + gap + lableWindowSize)))
            indx += 1
            # if (indx == maxIndx):
            #     break
            transactionGraph = nx.MultiDiGraph()

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=windowSize)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(
                days=gap) + dt.timedelta(
                days=lableWindowSize)
            label_start_date = window_start_date + dt.timedelta(days=windowSize) + dt.timedelta(days=gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 1 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 0

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

            timeWindowSequence = self.process_TDA_extracted_rnn_sequence(selectedNetworkInGraphDataWindow, node_features)

            # timeWindowSequenceRaw = self.processRawExtractedRnnSequence(selectedNetworkInGraphDataWindow, node_features)
            # result_list = []
            # first_key_tda = next(iter(timeWindowSequence))
            # tda_value = timeWindowSequence[first_key_tda]
            #
            # first_key_raw = next(iter(timeWindowSequenceRaw))
            # raw_value = timeWindowSequenceRaw[first_key_raw]
            #
            # for sublist1, sublist2 in zip(tda_value, raw_value):
            #     merged_sublist = sublist1 + sublist2
            #     result_list.append(merged_sublist)
            #
            #
            # totalRnnSequenceData.append({first_key_tda + "_" + first_key_raw : result_list})

            totalRnnSequenceData.append(timeWindowSequence)
            totalRnnLabelData.append(label)
            window_start_date = window_start_date + dt.timedelta(days=1)

        total_merged_seq = self.merge_dicts(totalRnnSequenceData)
        finalDict = {"sequence": total_merged_seq, "label": totalRnnLabelData}
        directory = 'Sequence/' + str(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + '/seq_tda_ablation.txt',
                  'wb') as file_in:
            pickle.dump(finalDict, file_in)
            file_in.close()
    
    
    def extract_tda_features(self, graph):
        pass 
    
    
    def extract_graph_features(self, graph):
        pass 
    
    
    def extract_node_features(self, graph):
        pass
    
    
    def create_TDA_graph(self, data, label, htmlPath="", timeWindow=0, network=""):
        try:
            #n_clusters, n_cubes, perc_overlap = get_TDA_params(data)
            perc_overlap = 0.3
            n_cubes = 2
            cls = 2 
            Xfilt = data
            Xfilt = Xfilt.drop(columns=['nodeID'])
            mapper = km.KeplerMapper()
            scaler = MinMaxScaler(feature_range=(0, 1))

            Xfilt = scaler.fit_transform(Xfilt)
            lens = mapper.fit_transform(Xfilt, projection=sklearn.manifold.TSNE())  # Try UMAP

            graph = mapper.map(
                lens,
                Xfilt,
                clusterer=sklearn.cluster.KMeans(n_clusters=cls, random_state=1618033),
                cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))  # 0.2 0.4

            # mapper.visualize(graph,
            #                  path_html=htmlPath + "/mapper_output_{}_day_{}_cubes_{}_overlap_{}.html".format(
            #                      network.split(".")[0], timeWindow, n_cube, overlap),
            #                  title="Mapper graph for network {} in Day {}".format(network.split(".")[0],
            #                                                                       timeWindow))


            # removing al the nodes without any edges (Just looking at the links)
            tdaGraph = nx.Graph()
            for key, value in graph['links'].items():
                tdaGraph.add_nodes_from([(key, {"cluster_size": len(graph["nodes"][key])})])
                for to_add in value:
                    tdaGraph.add_nodes_from([(to_add, {"cluster_size": len(graph["nodes"][to_add])})])
                    tdaGraph.add_edge(key, to_add)

            # we have the tda Graph here
            # convert TDA graph to pytorch data
            directory = 'PygGraphs/TimeSeries/' + network + '/TDA_Tuned/Overlap_{}_Ncube_{}/'.format(perc_overlap,
                                                                                                        n_cubes)
            featureNames = ["cluster_size"]
            if not os.path.exists(directory):
                os.makedirs(directory)
            pygData = self.from_networkx(tdaGraph, label=label, group_node_attrs=featureNames)
            with open(directory + "/" + network + "_" + "TDA_graph(cube-{},overlap-{})_".format(n_cubes,
                                                                                                perc_overlap) + str(
                timeWindow), 'wb') as f:
                pickle.dump(pygData, f)

        except Exception as e:
            print(str(e))
    
    
    def tda_process(self):
        pass 
    
    
    def process_TDA_extracted_rnn_sequence(self):
        pass
