

from matplotlib import pyplot as plt
import numpy as np

ep = 20e-3
a = 1
x0 = 0
x1 = 1
N = 80

# boundary values
A = 0
B = 1

x = np.linspace(start=x0, stop=x1, num=N)
dx = x[2]-x[1]
print(f"dx: {dx}")

y = (1 + a) * ( 1/(1+a*x) - np.exp(-x/ep) )

M = np.zeros((N-2,N-2), dtype=float)

w = 1 - (dx*a)/(1+a*x)
v = - ep/((1+a*x)*dx)

diag_main = w[1:-1] - 2*v[1:-1]*w[1:-1]
diag_down = w[1:-2]*v[1:-2]
diag_up = w[2:-1]*v[2:-1]-1
np.fill_diagonal(M, diag_main)
np.fill_diagonal(M[1:, :-1], diag_down)  # suprime la primera fila y la ultima col para sacar la diag inferior
np.fill_diagonal(M[:-1, 1:], diag_up)  # suprime la ultima fila y la primera col para sacar la diag superior

b = np.zeros((N-2, 1), dtype=float)
b[0] = A*v[0]*w[0]
b[-1] = B*(v[-1]*w[-1]-1)


yd = np.linalg.solve(M, -b)
yd = np.append(yd, B)
yd = np.insert(yd, 0, A)

plt.plot(x, y, 'b-o', label='real solution')
plt.plot(x, yd, 'g-o', label='discrete solution')

plt.title('y(x)')
plt.grid(True)
plt.xlabel('(x)')
plt.ylabel('')
plt.legend(loc='best')
plt.show()


