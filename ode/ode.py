import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


t = np.linspace( -2, 2, 51) # vector of time
y0 = 0 # start value
y = odeint(lambda y, t: 2*t , y0, t) # solve eq.
y = np.array(y).flatten() 
plt.plot( t, y,'-sr', linewidth=3) # graphic
plt.show() # display
