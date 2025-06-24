
"""
i'm comparing the real solution of 2 concentrical conducting spheres of
radius a and b where a<b and b is a 0V and a is a Va volts. this is a boundary value problem
but i'm using it to test my implementation of a solution by V(r+dr) = exp[dr*Operator]*V(r)
"""

from matplotlib import pyplot as plt
import numpy as np


a = 10
b = 20
Va = 100
N = 10

r = np.linspace(start=a, stop=b, num=N)
dr = r[2]-r[1]

V = Va * (a*(b-r))/(r*(b-a))
print(V[1])

Vd = np.zeros(N)
Vd[0] = V[0]
Vd[1] = V[1]


def next_step_1st_order(v_n_1, v_n, dr, r):
    return ( v_n_1*(1+(r/dr)) - v_n*(r/(2*dr)) )/( 1+(r/(2*dr)) )

def next_step_2nd_order(v_n_1, v_n, dr, r):
    return ( v_n_1*(1+(r/dr)) - v_n*(r/(2*dr)) )/( 1+(r/(2*dr)) )


for n in range(2, len(r)):
    Vd[n] = next_step_1st_order(Vd[n-1], Vd[n-2], dr, r[n-1])


plt.plot(r, V, 'b-o', label='real solution')
plt.plot(r, Vd, 'g-o', label='discrete solution')

plt.title('Potential vs. Radius')
plt.grid(True)
plt.xlabel('Radius (r)')
plt.ylabel('Voltage (V)')
plt.legend(loc='best')
plt.show()

