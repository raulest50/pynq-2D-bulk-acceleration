

import numpy as np
from matplotlib import pyplot as plt

N = 200
t = np.linspace(start=0, stop= 10, num=N)
dt = t[2]-t[1]
A = 0
B = np.pi/4
g = 9.8

M = np.zeros((N-2, N-2), dtype=float )
np.fill_diagonal(M, (g*dt**2 - 2))
np.fill_diagonal(M[1:, :-1], 1)  # suprime la primera fila y la ultima col para sacar la diag inferior
np.fill_diagonal(M[:-1, 1:], 1)

b = np.zeros((N-2, 1), dtype=float)
b[0] = 0
b[-1] = B

yd = np.linalg.solve(M, -b)
yd = np.append(yd, B)
yd = np.insert(yd, 0, A)

plt.plot(t, yd, 'b-o', label='numerical solution')

plt.title('T(t)')
plt.grid(True)
plt.xlabel('(t)')
plt.ylabel('Angle')
plt.legend(loc='best')
plt.show()

