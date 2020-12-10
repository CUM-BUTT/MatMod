from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

"""
a1 скорость наращивания вооружений первой страной;
a2 скорость наращивания вооружений второй страной;
b1 скорость устаревания вооружений первой страны;
b2 скорость устаревания вооружений второй страны;
m1 начальный объем вооружений первой страны;
m2 начальный объем вооружений второй страны;
c1 уровень недоверия первой страны;
c2 уровень недоверия второй страны.
"""
class Model:
    def __init__(self, a1, a2, b1, b2, c1, c2):
        self.dt = 0.01
        self.time = 0

        self.a1, self.b1, self.c1, \
        self.a2, self.b2, self.c2 = \
            a1, b1, c1, \
            a2, b2, c2

        m1, m2 = 0.6, 0.9
        self.m1 = np.array([m1 * 1 for __ in enumerate(self.a1)], dtype=float)
        self.m2 = np.array([m2 * 1 for __ in enumerate(self.a2)], dtype=float)
        # system at every step
        self.story = []


    def SaveStory(self):
        self.story += [[self.m1.copy(), self.m2.copy()]]

    def Iteration(self):
        self.time += self.dt
        m1, m2 = self.m1.copy(), self.m2.copy()

        self.m1 += (self.a1 * m2 - self.b1 * m1 + self.c1) * self.dt
        self.m2 += (self.a2 * m1 - self.b2 * m2 + self.c2) * self.dt

    def Run(self, iteration_count=100):  # Run model
        for t in range(iteration_count):
            self.Iteration()
            self.SaveStory()
        self.story = np.rot90(self.story)
        print('iterations end')

    def MakePlot(self):
        x = self.story[0]
        y = self.story[1]

        plt.axis((0, 2, 0, 2))
        plt.plot(x, y, )
        plt.show()

    def MakePoints(self):
        max, eps = 1000, 0.0001
        finals = []
        for x, y in zip(self.m1, self.m2):
            #infinit race
            if (x > max and y > max):
                color = 'r'
            # no ammo
            elif (x < eps and y < eps):
                color = 'b'
            # static point
            elif (x <= max and y <= max):
                color = 'y'



            finals += color

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_zlim3d(0, 10000)

        #x = a*a y=b*b z=m + m
        ax.scatter(self.a1*self.a2, self.b1 * self.b2, self.m1+self.m2, c=finals, s=3, alpha=0.2)
        plt.savefig('points')
        plt.show()

def GetCommonSituations():
    # стационарная точка для любых a0*a1 < b0*b1
    m = Model([1], [0.1], [20], [3], [0.5], [0.6])
    m.Run(1000)
    m.MakePlot()

    # бесконечная гонка для любых a0*a1 > b0*b1
    m = Model([2], [4], [2], [3], [3], [0.6])
    m.Run(1000)
    m.MakePlot()

    # 0 для любых a0*a1 < b0*b1 и с0 = с1 = 0
    m = Model([1], [0.1], [2], [3], [0], [0])
    m.Run(1000)
    m.MakePlot()
    plt.savefig('common_situations.png')

    plt.show()

def GetAllSituations():
    step = 0.4
    start, stop = 0, 2

    probabilities = np.linspace(start, stop, int(stop / step))
    combinations = product(probabilities, repeat=6)

    combinations = np.array(list(combinations), dtype=float)
    combinations = np.rot90(combinations)

    model = Model(*combinations)
    model.Run(1000)

    return model

model = GetAllSituations()
model.MakePlot()
model.MakePoints()






