import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Model:
    def __init__(self):
        self.dt = 0.1
        self.time = 0

        self.useful_mass = 100
        self.oil_mass = 1000

        self.speed = np.array([0, 0], dtype=float)
        self.position = np.array([0, 0], dtype=float)
        self.density_table = pd.read_csv('density.csv', delimiter=';', dtype=float)
        self.story = pd.DataFrame(
            columns=['time', 'position', 'speed', 'mass'],
            dtype=float)  # состояние системы на каждом шаге

    def GetMass(self):
        return self.useful_mass + self.oil_mass

    def ResistanceForce(self):
        p = self.density_table[self.density_table.height <= self.position[1]].density.iloc[-1]
        # frontal square
        s = 9 * 10 ** 0
        c = 1
        v = self.speed.copy()
        return -p * v * v * s * c * 0.5

    def Gravitation(self):
        g = 9.80665
        return -np.array([0, g * self.GetMass()], dtype=float)

    def EnginePush(self):
        if (self.oil_mass > 0):
            delta_speed, delta_mass = 1.1, 10
            self.oil_mass -= delta_mass * self.dt
            return delta_speed * self.GetMass() * np.array([0.1, 1], dtype=float) / self.dt
        else:
            return 0

    def SaveStory(self):
        self.story.loc[len(self.story)] = [self.time, self.position.copy(),
                                           self.speed.copy(), self.GetMass()]

    def Iteration(self):
        forces = self.ResistanceForce() + self.Gravitation() + self.EnginePush()
        self.speed += forces / (self.GetMass())
        self.position += self.speed.copy() * self.dt
        self.time += self.dt

    def CalculateImpulsAndEnergy(self):
        self.story['impulse'] = self.story.mass * self.story.speed
        self.story['energy'] = self.story.mass * self.story['speed'].apply(lambda x: np.linalg.norm(x) ** 2)


    def Run(self, iteration_count=100):  # Run model
        for __ in range(iteration_count):
            self.Iteration()
            self.SaveStory()

        self.CalculateImpulsAndEnergy()


    def MakePlot(self):
        t = self.story.time
        y = [x[1] for x in self.story.position]
        v = [x[1] for x in self.story.speed]
        m = self.story.mass
        fig, ax = plt.subplots(nrows=3, ncols=1)

        ax[0].plot(t, y, label='position')
        ax[1].plot(t, m, label='mass')
        ax[2].plot(t, v, label='speed')

        [a.legend() for a in ax]
        plt.show()
        fig.savefig('plot.png')

    def SavePlotAsMd(self):
        open('plot.md', mode='a').write('![plot](plot.png)')

m = Model()
m.Run()
m.MakePlot()
m.SavePlotAsMd()