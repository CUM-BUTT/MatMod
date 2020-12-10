import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        self.dt = 0.01
        self.time = 0

        self.mass = 0.01
        self.speed = np.array([900, 0], dtype=float)
        self.position = np.array([0, 1], dtype=float)

        self.story = pd.DataFrame(
            columns=['time', 'position', 'speed'],
            dtype=float)  # состояние системы на каждом шаге

    def ResistanceForce(self):
        p = 1.2
        # frontal square
        s = 10 ** -3
        c = 1
        v = self.speed.copy()
        return -(p * s * c * v ** 2) * 0.5

    def Gravitation(self):
        g = 9.80665
        return -np.array([0, g * self.mass], dtype=float)

    def SaveStory(self):
        self.story.loc[len(self.story)] = [self.time, self.position.copy(),
                                           self.speed.copy()]

    def Iteration(self):
        forces = self.ResistanceForce() + self.Gravitation()
        self.speed += forces #/ self.mass
        self.position += self.speed.copy() * self.dt
        self.time += self.dt

    def CalculateImpulsAndEnergy(self):
        self.story['impulse'] = self.mass * self.story.speed
        self.story['energy'] = self.mass * self.story['speed'].apply(lambda x: np.linalg.norm(x) ** 2)


    def Run(self, iteration_count=100):  # Run model
        while self.position[1] > 0:
            self.Iteration()
            self.SaveStory()

        self.CalculateImpulsAndEnergy()


    def MakePlot(self):
        t = self.story.time

        y = [x[1] for x in self.story.position]
        x = [x[0] for x in self.story.position]
        v_abs = [np.linalg.norm(x) for x in self.story.speed]
        fig, ax = plt.subplots(nrows=2, ncols=1)

        ax[0].plot(x, y, label='position')
        ax[1].plot(t, v_abs, label='speed')

        [a.legend() for a in ax]
        plt.show()
        fig.savefig('plot.png')


m = Model()
m.Run()
m.MakePlot()
