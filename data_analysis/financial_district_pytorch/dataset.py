import numpy as np
import scipy.io as spio
import pandas as pd

class trafficDataLoader():
    """
    Load traffic data from file. [graph, time series]
    """
    def __init__(self, taskID, finterval):
        self.graph = self.graphLoader(taskID)
        self.A = self.formAdjMatrix(self.graph)
        self.data = self.seriesLoader(finterval)
        self.nNode = len(self.graph)

    def graphLoader(self, taskID):
        graph = []
        with open('data/graph_%d.csv' % taskID, 'r') as f:
            for line in f:
                outEdge = line[:-1].split(' ')
                outEdge = outEdge[1:]
                for i in range(len(outEdge)):
                    outEdge[i] = int(outEdge[i])
                graph.append(outEdge)

        return graph

    def seriesLoader(self, finterval):
        # create dataframe
        data_df = pd.read_csv('data/financial_district_%d_knn.csv' % finterval, dtype={'avg_speed':'float64', 'edge_id':'int64'}, 
                      converters={'pub_millis':pd.to_datetime}, names=['pub_millis', 'edge_id', 'avg_speed'])

        # aggregate df by edge id and create a list of speeds for each edge id
        speed_list = data_df.groupby(['edge_id'])['avg_speed'].apply(list)

        # store each list of speeds into a new list
        data = []
        for i in range(len(speed_list)):
            data.append(speed_list.iloc[i])
        data = np.array(data)

        # update this later to dynamically change with how many input features we have
        self.dimFeature = 1

        return data

    def formAdjMatrix(self, graph):
        dim = len(graph)
        A = np.zeros([dim, dim])
        for i in range(dim):
            for j in graph[i]:
                A[i, j] = 1
        return A
