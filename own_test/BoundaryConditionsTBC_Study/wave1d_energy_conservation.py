import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from own_test.MyHelperFunctions import u_gaussian, ut_gaussian, create_tridiagonal_matrix

# ——— Domain & parameters ———
c  = 4.0             # wave speed
ti, tf = 0.0, 2.0    # time interval
xi, xf = 0.0, 10.0   # spatial interval

Nt = 320             # number of time steps
Nx = 160             # number of space steps
x  = np.linspace(xi, xf, Nx)
t  = np.linspace(ti, tf, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# CFL‐like parameter for CN scheme
alpha = (c * dt)**2 / 2

# ——— Initial data ———
f = u_gaussian(x=x, t=0, σ=0.3, c=c)   # u(x,0)
g = ut_gaussian(x=x, t=0, σ=0.3, c=c)  # u_t(x,0)

# ——— Discrete Laplacian & identity ———
D2x = create_tridiagonal_matrix(n=Nx, md=-2/dx**2, od=1/dx**2)
I   = create_tridiagonal_matrix(n=Nx, md=1.0,   od=0.0)

# ——— Build the CN‐system matrix ———
A = I - alpha * D2x   # (I - (cΔt)^2/2 Dxx)

# ——— Second‐order accurate “ghost” layer u^{-1} ———
y_nm1 = f - dt * g + alpha * (D2x.dot(f))
y_n   = f.copy()      # u^0

# ——— Plot setup ———
fig, ax = plt.subplots()
line, = ax.plot(x, y_n, lw=2)
ax.set_xlim(xi, xf)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('1D Wave (Crank–Nicolson)')
ax.grid(True, which='major')
ax.grid(True, which='minor', alpha=0.2)
ax.minorticks_on()


# ——— Animation function ———
def animate(frame):
    global y_nm1, y_n
    # RHS: 2u^n - u^{n-1} + α D2x u^{n-1}
    b = 2*y_n - y_nm1 + alpha * (D2x.dot(y_nm1))
    # solve for u^{n+1}
    y_np1 = np.linalg.solve(A, b)
    # enforce fixed ends
    y_np1[0] = 0.0
    y_np1[-1] = 0.0

    # update plot
    line.set_ydata(y_np1)
    ax.set_title(f'1D Wave (CN) — t = {t[frame]:.3f}s')

    # shift time‐levels
    y_nm1, y_n = y_n, y_np1
    return line,

# ——— Run animation ———
ani = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)
plt.show()
