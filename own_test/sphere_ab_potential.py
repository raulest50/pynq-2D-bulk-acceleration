

from matplotlib import pyplot as plt
from numpy import linspace


a = 10
b = 20
Va = 100

r = linspace(start=a, stop=b, num=10)

V = Va * (a*(b-r))/(r*(b-a))

plt.plot(r, V, '-o')
plt.show()
