import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

def GetSpeed(height):
    bul_m, bag_m, v, t0, t1, c, h, g = sym.symbols('bul_m bag_m v t0 t1 c h g')
    f = bul_m*v*v - (t1 - t0)*c*bul_m - bag_m*g*h
    v = sym.solveset(f, v)
    v = v.subs([(h, height), (t0, 20), (t1, 327), (c, 130), (g, 9.880665), (bul_m, 0.001), (bag_m, 1000)])
    return (v.args[1])

def RunTests(test_count):
    mu, sigma= 0.0601, 0.01 # mean and standard deviation
    heights = np.random.normal(mu, sigma, test_count)
    speeds = np.array([GetSpeed(h) for h in heights], dtype=float)

    fig, ax = plt.subplots(nrows=2, ncols=1)


    ax[0].hist(heights, bins=int(test_count*0.1),
               density=False, color='g', label='heights')
    ax[1].hist(speeds, bins=int(test_count*0.1),
               density=False, color='b', label='speeds')

    [a.legend() for a in ax]
    plt.show()

RunTests(100)

