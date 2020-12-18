from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import scipy
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from matplotlib import pyplot as plt


class Model:
    def __init__(self, pos, v, m):
        self.dt = 0.01
        self.time = 0
        self.v0 = np.array(v, dtype=float)
        self.v = np.array(v, dtype=float)
        self.degree = 3

        self.story = []

        G = 1
        m = np.array(m, dtype=float)
        m = [[[mi*mj]*2 for j, mj in enumerate(m)] for i, mi in enumerate(m)]
        self.m = np.array(m, dtype=float) * G
        self.pos = np.array(pos, dtype=float)

    def SaveStory(self):
        self.story += [[self.pos[0].copy(), self.pos[1].copy()]]

    def Iteration(self):
        vectors = np.array([[u-v for v in self.pos]
                   for u in self.pos], dtype=float)
        magnitudes = np.array([[[np.linalg.norm(item, axis=0)]*2
                                for j, item in enumerate(row)]
                                for i, row in enumerate(vectors)], dtype=float)

        dv = self.m * vectors * (magnitudes ** -self.degree)
        for i, dvi in enumerate(dv):
            dv[i, i] = np.zeros(shape=dv[i, i].shape)
        dv = np.sum(dv, axis=0)
        self.v += dv * self.dt
        self.pos += self.v * self.dt

    def Run(self, iteration_count=100):  # Run model
        for t in range(iteration_count):
            self.SaveStory()
            self.Iteration()
        self.story = np.array(self.story)
        print('iterations end')

    def MakePlot(self):
        story = np.rot90(self.story)
        layer_num = 12
        x = story[0, :, 0, :]
        y = story[0, :, 1, :]

        plt.axis((0, 10, 0, 10))
        plt.plot(x, y, )

    def MakePoints(self):
        story = np.rot90(self.story)

        x = story[0, :, 0, -1]
        y = story[0, :, 1, -1]
        max, eps = 100, 0.0001
        finals = []
        for i, item in enumerate(zip(x, y)):
            xi, yi = item
            # out of border
            if (0 > xi > max and 0 > yi > max):
                color = 'r'
            # in border
            else:
                color = 'b'

            finals += color

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim3d(0, 10000)

        # x = a*a y=b*b z=m + m
        v0, v1 = self.v0[:, 0], self.v0[:, 1]
        ax.scatter(v0, v1,
                   c=finals, s=3, alpha=0.2)
        plt.savefig('points')

def GetAllSituations():
    step = 0.4
    start, stop = 0, 3

    probabilities = np.linspace(start, stop, int(stop / step))
    combinations = product(probabilities, repeat=4)

    combinations = np.array(list(combinations), dtype=float)
    combinations = np.rot90(combinations)

    a, b, c, d = combinations
    n = len(a)

    model = Model(pos=[[[1]*n, [1]*n], [[1]*n, [2]*n], ],
                  m=[[10]*n, [1]*n ],
                  v=[[a, b], [c, d], ])

    return model




m = GetAllSituations()
m.Run(100)

m.MakePlot()
plt.show()

m.MakePoints()
plt.show()
"""
a = np.ones(shape=(3, 3, 2))*10
b = np.ones(shape=(3, 3))
print(a * b)
"""