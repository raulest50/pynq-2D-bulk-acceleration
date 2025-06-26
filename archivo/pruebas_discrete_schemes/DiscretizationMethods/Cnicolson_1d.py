
import numpy as np
from matplotlib import pyplot as plt


D = 0.5
L = 10
Nm = 100
x = np.linspace(-L, L, Nm, dtype=float)
dx = x[2]-x[1]
d0 = np.sin(np.pi*(x+10)/20)
tf = 20
Nn = 500
t = np.linspace(0, tf, Nn, dtype=float)
dt = t[2]-t[1]

d_evo = d0*np.exp(-D*(np.pi/20)**2*tf)

c = dt*D/(2*dx**2)
cc = (1-2*c)
cm = (-1-2*c)

M = np.zeros((Nm, Nm), dtype=float)

np.fill_diagonal(M, cm)
np.fill_diagonal(M[1:, :-1], c)  # suprime la primera fila y la ultima col para sacar la diag inferior
np.fill_diagonal(M[:-1, 1:], c)  # suprime la ultima fila y la primera col para sacar la diag superior

B = np.zeros((Nm, Nm + 2))
np.fill_diagonal(B[:, 1:-1], cc)
np.fill_diagonal(B[:, 2:], c)
np.fill_diagonal(B[:, :-2], c)

def one_step(dn):
    dn_aux = dn
    dn_aux = np.append(dn_aux, 0)
    dn_aux = np.insert(dn_aux, 0, 0)
    b = B.dot(dn_aux)
    return np.linalg.solve(M, -b)

dn = d0
for st in t:
    dn = one_step(dn)


plt.title('Initial and final concentration')
plt.ylabel('concentration')
plt.xlabel('position x')
plt.grid(True)


plt.plot(x, d0, 'b-', label='initial concentration')
plt.plot(x, d_evo, 'm-', label=f'time evolution at tf = {tf}')
plt.plot(x, dn, 'c-o', label=f'time evolution numeric solution, tf = {tf}')

plt.legend(loc='best')
plt.show()




