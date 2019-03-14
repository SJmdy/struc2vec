import networkx as nx
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pandas as pd
from six.moves import cPickle as pickle
import os


class Graph:
    def __init__(self, file_path: str, directed: bool, weighted: bool, diameter, reset: bool):
        self.file_path = file_path
        self.directed = directed
        self.weighted = weighted
        self.diameter = diameter
        self.G = None
        self.k_nbrs = None
        self.s_nbrs = None
        self.in_nbrs = None
        self.reset = reset

        self.build_graph()
        self.check()
        self.get_k_neighbors()
        self.get_sequence_degree()
        # self.get_in_nbrs()

    def build_graph(self):
        if self.weighted:
            self.G = nx.DiGraph(
                nx.read_edgelist(path=self.file_path, create_using=nx.DiGraph(), nodetype=int, data=(('weight', float),)))
            self.G = self.normalize_weight()
        else:
            self.G = nx.DiGraph(nx.read_edgelist(path=self.file_path, create_using=nx.DiGraph(), nodetype=int))

        if not self.directed:
            self.G = self.G.to_undirected()

    def check(self):
        if len(self.G.nodes) == max(self.G.nodes):
            r = pd.read_csv(filepath_or_buffer=self.file_path, sep=' ', header=None, names=['u', 'v', 'w'])
            r['u'] -= 1
            r['v'] -= 1
            r.to_csv(path_or_buf=self.file_path, sep=' ', header=False, index=False)
            del self.G
            self.build_graph()

    def normalize_weight(self):
        print('normalize weight start...')
        for node, node_neighboors in self.G.adj.items():
            sum_weights = sum([node_neighboors[i]['weight'] for i in node_neighboors])
            for nbr in node_neighboors:
                node_neighboors[nbr]['weight'] /= sum_weights
        print('normalize weight finish...')
        return self.G

    """
    k_nbrs : {
        k : {
            node1: neighbors,
            node2: neighbors
        }
    }
    """

    def get_k_neighbors(self):
        print('get k_nbrs start...')

        # pickle_file = self.file_path + '.k_nbrs.pickle'
        # if os.path.exists(pickle_file) and not self.reset:
        #     self.k_nbrs = self.load(pickle_file)
        #     print('get k_nbrs finish...')
        #     return

        k_nbrs = {}
        # self.G = nx.Graph()
        for k in range(0, self.diameter):
            k_nbrs[k] = {}
            if k == 0:
                for n in self.G.nodes:
                    k_nbrs[k][n] = list(self.G.neighbors(n))
            else:
                for n in k_nbrs[k - 1]:
                    nbrs = k_nbrs[k - 1][n]
                    k_nbrs[k][n] = []
                    for nbr in nbrs:
                        k_nbrs[k][n] += list(self.G.neighbors(nbr))
                    # k_nbrs[k][n] = [item for sublist in [list(self.G.neighbors(i)) for i in nbrs] for item in sublist]
        self.k_nbrs = k_nbrs
        print('get k_nbrs finish...')
        # self.save(content=k_nbrs, file_path=pickle_file)

    """
    s_nbrs : {
        k: {
            node1: sequence1,
            node2: sequence2
        }
    }
    """

    def get_sequence_degree(self):
        print('get s_nbrs start...')

        # pickle_file = self.file_path + '.s_nbrs.pickle'
        # if os.path.exists(pickle_file) and not self.reset:
        #     self.s_nbrs = self.load(pickle_file)
        #     print('get s_nbrs finish...')
        #     return

        s_nbrs = {}
        if self.directed:
            for k in range(0, self.diameter):
                s_nbrs[k] = {}
                for n in self.k_nbrs[k]:
                    s_nbrs[k][n] = [self.G.out_degree(i) for i in self.k_nbrs[k][n]]
                    s_nbrs[k][n].sort()
        else:
            for k in range(0, self.diameter):
                s_nbrs[k] = {}
                for n in self.k_nbrs[k]:
                    s_nbrs[k][n] = [self.G.degree(i) for i in self.k_nbrs[k][n]]
                    s_nbrs[k][n].sort()
        self.s_nbrs = s_nbrs
        print('get s_nbrs finish')
        # self.save(self.s_nbrs, file_path=pickle_file)

    def get_in_nbrs(self):
        in_nbrs = {}
        for u in self.G.nodes:
            in_nbrs[u] = [i[0] for i in self.G.in_edges(u)]
        self.in_nbrs = in_nbrs

    @staticmethod
    def get_dtw_distance(s1, s2):
        distance, _ = fastdtw(s1, s2, dist=euclidean)
        return distance

    @staticmethod
    def save(content, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(content, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            save = pickle.load(f)
        return save
