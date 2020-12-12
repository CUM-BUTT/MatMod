from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import scipy
from scipy import spatial
from matplotlib import pyplot as plt

class Model:
    def __init__(self, pos, v, m):
        self.dt = 0.01
        self.time = 0
        self.v = np.array(v)
        self.degree = 3

        self.story = []

        G = 1
        m = np.outer(m, m) * G
        np.fill_diagonal(m, 0)
        self.m = np.array([[ [mj, mj] for j, mj in enumerate(mi)]
                                for i, mi in enumerate(m)])

        self.zero_diag = np.array([[[mj, mj] for j, mj in enumerate(mi)]
                           for i, mi in enumerate(m)])

        self.pos = np.array(pos)

    def SaveStory(self):
        pass
        #self.story += [self.pos.copy()]

    def Iteration(self):
        self.time += self.dt
        vectors = np.array([[ np.abs(u - v)
                                for j, v in enumerate(self.pos)]
                                for i, u in enumerate(self.pos)])
        #magnitudes = np.linalg.norm(vectors, axis=2)
        #vectors = vectors.

        dv = self.m / (vectors ** self.degree)
        dv = np.sum(dv, axis=(0))
        self.v += dv * self.dt
        self.pos += self.v * self.dt

    def Run(self, iteration_count=100):  # Run model
        for t in range(iteration_count):
            self.SaveStory()
            self.Iteration()

        print('iterations end')

    def MakePlot(self):
        x = self.story[0]
        y = self.story[1]

        plt.axis((0, 100, 0, 100))
        plt.plot(x, y, )

    def MakePoints(self):
        pass




def GetAllSituations():
    step = 0.1
    start, stop = 0, 2

    probabilities = np.linspace(start, stop, int(stop / step))
    combinations = product(probabilities, repeat=4)

    combinations = np.array(list(combinations), dtype=float)
    combinations = np.rot90(combinations)

    model = Model(*combinations)
    model.Run(1000)

    return model


"""
pos = np.array([[1,0],[0,1],[3,1],[3,5]])
distances = np.array([[np.abs(u - v)
                       for j, v in enumerate(pos) ]
                       for i, u in enumerate(pos)])
print(distances)
magnitudes = np.linalg.norm(distances, axis=(2))
print(magnitudes)
"""

m = Model(pos=[[1, 0], [0, 1], [3, 1], [3, 5]], m=[1, 2, 3, 4], v=[[1, 0], [0, 1], [3, 1], [3, 5]])
m.Run()