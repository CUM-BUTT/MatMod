import numpy as np
from matplotlib import pyplot as plt

class SpaceBody:
    story = []
    dt = 0.01
    degree = -2

    def __init__(self, m, pos, v):
        self.v = np.array(v, dtype=float)
        self.m = np.array(m, dtype=float)
        self.pos = np.array(pos, dtype=float)

    def SetAnothers(self, anothers):
        self.anothers = anothers

    def SaveStory(self):
        self.story += [self.pos.copy()]

    def Interation(self, another):
        distance = (another.pos - self.pos)
        magnitude = np.linalg.norm(distance)
        if magnitude > 0.01:
            dv = (another.m * self.m)*(magnitude**self.degree)
            self.v += dv * self.dt
            self.pos += self.v * self.dt

    def Iteration(self):
        for another in self.anothers:
            self.Interation(another)

        self.SaveStory()


    def MakePlot(self):
        self.story = np.rot90(self.story)
        x = self.story[0]
        y = self.story[1]

        plt.plot(x, y)


class Space:
    bodies = []
    def __init__(self):
        self.bodies = [SpaceBody(m=1, pos=[1, 0], v=[0,0]),
                  SpaceBody(m=1, pos=[0, 1], v=[0, 0])]

        [body.SetAnothers(self.bodies) for body in self.bodies]

    def Run(self, iteration_count = 1000):
        for i in range(iteration_count):
            for b in self.bodies:
                b.Iteration()

        print('iteration end')

        for b in self.bodies:
            b.MakePlot()

        plt.show()

s = Space()
s.Run()