
import numpy as np
from matplotlib import pyplot as plt


D = 0.5
L = 10
N = 25
x = np.linspace(-L, L, N, dtype=float)
dx = x[2]-x[1]
d0 = np.sin(np.pi*(x+10)/20)
tf = 20

d_evo = d0*np.exp(-D*(np.pi/20)**2*tf)

plt.title('Initial and final concentration')
plt.ylabel('concentration')
plt.xlabel('position x')
plt.grid(True)


plt.plot(x, d0, 'b-o'
                '', label='initial concentration')
plt.plot(x, d_evo, 'm-o', label=f'time evolution at tf = {tf}')

plt.legend(loc='best')
plt.show()




