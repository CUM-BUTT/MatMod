from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt



class Model:
    def __init__(self, prey_fertility, predator_kill, predator_death, predator_fertility):
        self.dt = 0.01
        self.time = 0

        self.a, self.b, self.c, self.d = prey_fertility, predator_kill, predator_death, predator_fertility
        # first is preys, second is predators count
        count = 10
        self.x = np.array([count for __ in enumerate(self.a)], dtype=float)
        self.y = np.array([count for __ in enumerate(self.a)], dtype=float)
        # system at every step
        self.story = []


    def SaveStory(self):
        self.story += [[self.x.copy(), self.y.copy()]]

    def Iteration(self):
        self.time += self.dt
        x, y = self.x.copy(), self.y.copy()
        self.x += (self.a - self.b * y) * x * self.dt
        self.y -= (self.c - self.d * x) * y * self.dt

    def Run(self, iteration_count=100):  # Run model
        for t in range(iteration_count):
            self.SaveStory()
            self.Iteration()

        self.story = np.rot90(self.story)
        print('iterations end')

    def MakePlot(self):
        x = self.story[0]
        y = self.story[1]

        plt.axis((0, 100, 0, 100))
        plt.plot(x, y, )

    def MakePoints(self):
        eps = 0.01
        colors = []
        for x, y in zip(self.x, self.y):

            if (x > eps and y > eps):
                color = 'g'
            # all dead
            elif (x <= eps and y <= eps):
                color = 'r'
            # prey alive
            else:
                color = 'b'
            colors += color


        plt.scatter(self.a, self.b * self.c * self.d,c=colors, s=3)



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

model = GetAllSituations()

model.MakePlot()
plt.show()

model.MakePoints()
plt.show()