import networkx as nx
import numpy as np
import pandas as pd
import os
import time

from struc2vec import *

if __name__ == '__main__':
    t1 = time.time()
    in_file = 'graph/barbell.edgelist'
    out_file = in_file.replace('graph/', 'emb/').replace('.edgelist', '.emb')

    parameters = pd.Series(
        {'input': in_file, 'output': out_file, 'directed': True, 'weighted': True, 'diameter': 10, 'dimensions': 128,
         'walk_length': 20, 'num_walks': 5, 'reset': True})

    print(parameters)

    # s = Struc2Vec(in_file=parameters.input, directed=parameters.directed, weighted=parameters.weighted,
    #               diameter=parameters.diameter, dimensions=parameters.dimensions, walk_length=parameters.walk_length,
    #               num_walks=parameters.num_walks, out_file=parameters.output, reset=parameters.reset)
    # s.get_embedding()
    get_embedding_list(out_file)
    t2 = time.time()
    print('Total use time: %fs' % (t2 - t1))
