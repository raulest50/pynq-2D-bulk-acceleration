import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from the helpers package
from helpers import u_gaussian, ut_gaussian, utt_gaussian, uttt_gaussian, create_tridiagonal_matrix

# Domain parameters
c=4

ti=0
tf=2

xi = 0
xf = 10

Nt = 320
Nx = 400
x = np.linspace(xi, xf, Nx)
t = np.linspace(ti, tf, Nt)
dx=x[1]-x[0]
dt=t[1]-t[0]

print(f"courant_lewis_number = {(c*dt)/dx}")


print( f"dt: {dt} \n dx: {dx} \n" )

fx = u_gaussian(x=x, t=0, σ=0.3, c=c) # Initial condition position
gx = ut_gaussian(x=x, t=0, σ=0.3, c=c) # Initial condition velocity (derivative)
ggx = utt_gaussian(x=x, t=0, σ=0.3, c=c)
gggx = uttt_gaussian(x=x, t=0, σ=0.3, c=c)

fig, ax = plt.subplots()
ax.set_xlabel('Posicion x')
ax.set_ylabel('Amplitud')
ax.set_title("Pulso gaussiano - 1D Wave equation")
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(-1, 11)
ax.grid(True, which='major')
ax.grid(True, which='minor', alpha=0.2)
ax.minorticks_on()
#ax.plot(x, fx, "b")
#ax.plot(x, gx, "r")
#plt.show()

line, = ax.plot(x, fx)

N = len(x)

D2x = create_tridiagonal_matrix(n=N, md=-2/dx**2, od=1/dx**2)
I = create_tridiagonal_matrix(n=N, md=1, od=0)
A = ((c**2*dt**2)/2)*D2x-I
B = ((c**2*dt**2)/2)*D2x+2*I

y_0 = fx + (-dt*gx) + ((-dt)**2/2)*ggx + ((-dt)**3/6)*gggx
y_1 = fx.copy()

def next_step(A, B, y1, y0):
    b = np.dot(-B, y1) + y0
    y2 = np.linalg.solve(A, b)
    return y2

def animation(i):
    global y_0, y_1
    y_2 = next_step(A, B, y_1, y_0)
    line.set_ydata(y_2)
    y_0 = y_1
    y_1 = y_2
    return line,

ani = FuncAnimation(fig, animation, frames=Nt, interval=50)
plt.show()
