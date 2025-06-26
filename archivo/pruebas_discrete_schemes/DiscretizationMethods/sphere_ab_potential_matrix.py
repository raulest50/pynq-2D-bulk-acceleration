

from matplotlib import pyplot as plt
import numpy as np


a = 10
b = 20
Va = 100
Vb = 0
N = 12

r = np.linspace(start=a, stop=b, num=N)
dr = r[2]-r[1]

V = Va * (a*(b-r))/(r*(b-a))

M = np.zeros((N-2, N-2), dtype=float)


diag_main = r[1:-1]/dr**2
diag_down = -r[2:-1]/(2*dr**2)
diag_up = -r[1:-2]/(2*dr**2)
np.fill_diagonal(M, diag_main)
np.fill_diagonal(M[1:, :-1], diag_down)  # suprime la primera fila y la ultima col para sacar la diag inferior
np.fill_diagonal(M[:-1, 1:], diag_up)  # suprime la ultima fila y la primera col para sacar la diag superior

I = np.identity(N-2, dtype=float)
O = (I + dr*M)

Ifw = np.zeros((N-2, N-2), dtype=float)
np.fill_diagonal(Ifw[:-1, 1:], -1)
O = O + Ifw

b = np.zeros((N-2, 1), dtype=float)
b[0] = (-r[1]/(2*dr**2))*Va*dr
b[-1] = (1-r[-1]/(2*dr**2))*Vb


Vd = np.linalg.solve(O, -b)
Vd = np.append(Vd, 0)
Vd = np.insert(Vd, 0, Va)



print(dr)
print(r)

plt.plot(r, V, 'b-o', label='real solution')
plt.plot(r, Vd, 'g-o', label='discrete solution')

plt.title('Potential vs. Radius')
plt.grid(True)
plt.xlabel('Radius (r)')
plt.ylabel('Voltage (V)')
plt.legend(loc='best')
plt.show()