import networkx as nx
import numpy as np
import pandas as pd
import os
import time

from struc2vec import *
from show import *

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
    t2 = time.time()
    print('Total use time: %fs' % (t2 - t1))

    class_key = {
        0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A', 6: 'A', 7: 'A', 8: 'A', 9: 'A', 10: 'B', 11: 'C', 12: 'D',
        13: 'E', 14: 'F', 15: 'G', 16: 'G', 17: 'F', 18: 'E', 19: 'D', 20: 'C', 21: 'B', 22: 'A', 23: 'A',
        24: 'A', 25: 'A', 26: 'A', 27: 'A', 28: 'A', 29: 'A'
    }

    shower = Show(in_file=out_file)
    shower.get_vector()
    shower.get_pca()
    shower.show(class_key=class_key, title='barbell', image_name='images/barbell.png')
