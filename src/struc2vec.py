import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, KernelPCA

from graph import *


class Struc2Vec:
    def __init__(self, in_file: str, directed: bool, weighted: bool, diameter: int, dimensions: int,
                 walk_length: int, num_walks: int, out_file: str, reset: bool):
        self.diameter = diameter
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.in_file = in_file
        self.out_file = out_file
        self.reset = reset

        self.G = Graph(file_path=in_file, directed=directed, weighted=weighted, diameter=diameter, reset=reset)
        self.k_nbrs = self.G.k_nbrs
        self.s_nbrs = self.G.s_nbrs
        self.in_nbrs = self.G.in_nbrs

        self.nodes = list(self.G.G.nodes)
        self.nodes_number = len(self.nodes)
        self.edges_number = len(self.G.G.edges)

        self.fk_uv = np.zeros(shape=[diameter, self.nodes_number, self.nodes_number])
        self.wk_uv = np.zeros(shape=[diameter, self.nodes_number, self.nodes_number])

        self.wku = np.zeros(shape=[diameter, self.nodes_number, 2])
        self.wk_average = np.zeros(shape=[diameter])
        self.tku = np.zeros(shape=[diameter, self.nodes_number])
        self.zku = np.zeros(shape=[diameter, self.nodes_number])
        self.pk_uv = np.zeros(shape=[diameter, self.nodes_number, self.nodes_number])
        self.pku = np.zeros(shape=[diameter, self.nodes_number, 2])

        self.all_paths = []

        self.get_fk_uv()
        self.get_wk_uv()
        self.get_wk_average()
        self.get_tku()
        self.get_wku()
        self.get_zku()
        self.get_pk_uv()
        self.get_pku()
        self.get_walks()

    def get_fk_uv(self):
        """
            s_nbrs: {
                k: {
                    node: the degree sequence of node
                }
            }
        """
        print('get fk_uv start...')
        for k in range(0, self.diameter):
            for u in self.nodes:
                for v in self.nodes:
                    # 只有当u、v都有第k跳的节点时，fk_uv[u][v]才有意义
                    if len(self.s_nbrs[k][u]) > 0 and len(self.s_nbrs[k][v]) > 0:
                        if u != v:
                            u_nbrs = self.s_nbrs[k][u]
                            v_nbrs = self.s_nbrs[k][v]
                            dist_uv = self.G.get_dtw_distance(u_nbrs, v_nbrs)
                            self.fk_uv[k][u][v] = self.fk_uv[k - 1][u][v] + dist_uv if k > 0 else dist_uv
                        else:
                            self.fk_uv[k][u][v] = 0
                    else:
                        self.fk_uv[k][u][v] = 10000
        print('get fk_uv finish...')

    def get_wk_uv(self):
        self.wk_uv = np.exp(-1 * self.fk_uv)

    def get_wk_average(self):
        print('get wk_average start...')
        for k in range(0, self.diameter):
            edges_number = (self.edges_number * (self.edges_number - 1)) / 2 if k > 0 else self.edges_number
            self.wk_average[k] = self.wk_uv[k].sum() / edges_number
        print('get wk_average finish...')

    def get_tku(self):
        for k in range(0, self.diameter):
            for u in self.nodes:
                self.tku[k][u] = np.size(self.wk_uv[k][u][self.wk_uv[k][u] > self.wk_average[k]])

    def get_wku(self):
        for k in range(0, self.diameter):
            for u in self.nodes:
                # wku[k][u][0]: w(uk, uk+1)
                # wku[k][u][1]: w(uk, uk-1)
                self.wku[k][u][0] = np.log10(self.tku[k][u] + np.e)
                self.wku[k][u][1] = 1

    def get_zku(self):
        for k in range(0, self.diameter):
            for u in self.nodes:
                # 若self.zku[k][u] = 0，当且仅当在self.wk_uv[k][u]中，只有wk_uv[u][u]有值
                self.zku[k][u] = self.wk_uv[k][u].sum() - 1
                # if self.zku[k][u] == 0:
                #     print(k, u, self.wk_uv[k][u])

    def get_pk_uv(self):
        print('get pk uv start...')
        for k in range(0, self.diameter):
            for u in self.nodes:
                for v in self.nodes:
                    self.pk_uv[k][u][v] = self.wk_uv[k][u][v] / self.zku[k][u]
                    if self.zku[k][u] == 0:
                        print(self.pk_uv[k][u][v])
        print('get pk uv finish...')

    def get_pku(self):
        for k in range(0, self.diameter):
            for u in self.nodes:
                # self.pku[k][u][0]: pk(uk, uk+1) = w(uk, uk+1) / (w(uk, uk+1) + w(uk, uk-1))
                # self.pku[k][u][1]: 1 - pk(uk, uk+1)
                self.pku[k][u][0] = self.wku[k][u][0] / (self.wku[k][u][0] + self.wku[k][u][1])
                self.pku[k][u][1] = 1 - self.pku[k][u][0]

    @staticmethod
    def alias_setup(probs):
        '''
        :param probs: 某个概率分布
        :return: Alias数组与Prob数组
        '''
        K = len(probs)  # K为类别数目
        Prob = np.zeros(K)  # 对应Prob数组：落在原类型的概率
        Alias = np.zeros(K, dtype=np.int)  # 对应Alias数组：每一列第二层的类型

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []  # 存储比1小的列
        larger = []  # 存储比1大的列

        for kk, prob in enumerate(probs):
            Prob[kk] = K * prob  # 概率（每个类别概率乘以K，使得总和为K）
            if Prob[kk] < 1.0:  # 然后分为两类：大于1的和小于1的
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that appropriately allocate
        # the larger outcomes over the overall uniform mixture.

        # 通过拼凑，将各个类别都凑为1
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            Alias[small] = large  # 填充Alias数组
            Prob[large] = Prob[large] - (1.0 - Prob[small])  # 将大的分到小的上

            if Prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        # print("Prob is :", Prob)
        # print("Alias is :", Alias)
        return Alias, Prob

    @staticmethod
    def alias_draw(Alias, Prob):
        '''
        :param J: Alias数组
        :param q: Prob数组
        :return:一次采样结果
        '''
        K = len(Alias)

        # Draw from the overall uniform mixture.
        kk = int(np.floor(np.random.rand() * K))  # 随机取一列

        # Draw from the binary mixture, either keeping the small one, or choosing the associated larger one.
        # 采样过程：随机取某一列k（即[1,4]的随机整数，再随机产生一个[0-1]的小数c，）
        # 如果Prob[kk]大于c，
        if np.random.rand() < Prob[kk]:  # 比较
            return kk
        else:
            return Alias[kk]

    def get_walks(self):
        for u in self.nodes:
            for i in range(0, self.num_walks):
                path = [str(u)]
                v = u
                k = 0
                # print('u = ', u)
                while len(path) < self.walk_length:
                    r = np.random.random()
                    # before each step, the random walk first decides if it will change layers or walk on the current layer
                    if r < 0.3:
                        # print('select next node...')
                        probs = self.pk_uv[k][v]
                        # print('probs = ', probs)
                        J, q = self.alias_setup(probs=probs)
                        v = self.alias_draw(J, q)
                        # print('v = ', v)
                        path.append(str(v))
                    else:
                        # print('decided change layers, layer = ', k)
                        r = np.random.random()
                        layer_up_prob = self.pku[k][v][0]
                        if r < layer_up_prob:
                            k = k + 1 if k < self.diameter - 1 else k
                        else:
                            k = k - 1 if k > 0 else 0
                        # print('up prob = %f, now k = %d' % (layer_up_prob, k))
                        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                self.all_paths.append(path)

    def get_embedding(self):
        # print(self.all_paths)
        model = Word2Vec(self.all_paths, size=self.dimensions, window=5, min_count=0, workers=4)
        model.wv.save_word2vec_format(self.out_file)
        print("Representations created.")
