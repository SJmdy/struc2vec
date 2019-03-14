from pyecharts import Scatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from pyecharts_snapshot.main import make_a_snapshot


class Show:
    def __init__(self, in_file: str):
        self.in_file = in_file
        self.point_vector = None
        self.show_coordinates = None

    def get_vector(self):
        with open(self.in_file, 'r') as f:
            content = f.readlines()[1:]

        point_vector = pd.DataFrame(columns=['point', 'vector'])
        for i in content:
            line = i.split(' ')
            point = int(line[0])
            vector = np.array(line[1:], dtype=np.float32).tolist()
            point_vector = point_vector.append({'point': point, 'vector': vector}, ignore_index=True)
            # print(point, vector)
            # print(point_vector)
        self.point_vector = point_vector

    def get_pca(self):
        pca = PCA(n_components=2)
        # print(self.point_vector['vector'].values.tolist())
        pca.fit(self.point_vector['vector'].values.tolist())
        data_pca = pca.fit_transform(self.point_vector['vector'].values.tolist())

        show_coordinates = pd.DataFrame(columns=['point', 'vector'])
        print(data_pca)
        for i in range(0, data_pca.shape[0]):
            point = self.point_vector.iloc[i]['point']
            vector = self.point_vector.iloc[i]['vector']
            show_coordinates = show_coordinates.append({'point': point, 'vector': vector}, ignore_index=True)
        self.show_coordinates = show_coordinates

    def show(self, class_key: dict, title: str, image_name):
        # print(self.show_coordinates)
        scatter = Scatter(title=title, background_color='#ffe')
        for i in range(0, self.show_coordinates.shape[0]):
            point = self.show_coordinates.iloc[i]['point']
            # print(point, class_key[point])
            coordinates = self.show_coordinates.iloc[i]['vector']
            # print(type(coordinates), type(class_key[point]))
            scatter.add(class_key[point], [coordinates[0]], [coordinates[1]])
        temp_path = 'html/render.html'
        scatter.render(path=temp_path)
        make_a_snapshot(temp_path, image_name)

