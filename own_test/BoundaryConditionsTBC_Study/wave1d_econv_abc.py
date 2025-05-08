import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from MyHelperFunctions import u_gaussian, ut_gaussian, create_tridiagonal_matrix

# ——— Domain & parameters ———
c     = 4.0             # wave speed
ti, tf= 0.0, 2.0        # time interval
xi, xf= 0.0, 10.0       # spatial interval

Nt    = 320             # number of time steps
Nx    = 160             # number of space steps
x     = np.linspace(xi, xf, Nx)
t     = np.linspace(ti, tf, Nt)
dx    = x[1] - x[0]
dt    = t[1] - t[0]

# CFL‐like parameter for CN scheme
alpha = (c * dt)**2 / 2

# ——— Initial data ———
f = u_gaussian(x=x, t=0, σ=0.3, c=c)   # u(x,0)
g = ut_gaussian(x=x, t=0, σ=0.3, c=c)  # u_t(x,0)

# ——— Discrete Laplacian & identity ———
D2x = create_tridiagonal_matrix(n=Nx, md=-2/dx**2, od=1/dx**2)
I   = create_tridiagonal_matrix(n=Nx, md=1.0,   od=0.0)

# ——— Build the CN‐system matrix for interior unknowns ———
# Extract the submatrix for i=1..Nx-2
D2x_int = D2x[1:-1, 1:-1]
I_int   = I[1:-1, 1:-1]
A_int   = I_int - alpha * D2x_int
# Pre-factor LU if you like, or just solve with np.linalg.solve

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
ax.set_title('1D Wave (CN) with Absorbing BCs')
ax.grid(True, which='both', alpha=0.3)
ax.minorticks_on()

# ——— Animation function ———
def animate(frame):
    global y_nm1, y_n

    # 1) Build RHS for interior nodes 1..Nx-2
    #    b_i = 2 u^n_i - u^{n-1}_i + α (D2x u^{n-1})_i
    Dx_y_nm1 = D2x.dot(y_nm1)
    b_int    = 2*y_n[1:-1] - y_nm1[1:-1] + alpha * Dx_y_nm1[1:-1]

    # 2) Solve interior CN system
    y_np1 = np.empty_like(y_n)
    y_np1[1:-1] = np.linalg.solve(A_int, b_int)

    # 3) Apply first‐order absorbing BC at left (i=0)
    y_np1[0] = y_n[0] - (c * dt / dx) * (y_n[1] - y_n[0])

    # 4) Apply absorbing BC at right (i=Nx-1)
    y_np1[-1] = y_n[-1] + (c * dt / dx) * (y_n[-2] - y_n[-1])

    # 5) Update plot & time levels
    line.set_ydata(y_np1)
    ax.set_title(f'1D Wave (CN + Absorbing BC) — t = {t[frame]:.3f}s')

    y_nm1, y_n = y_n, y_np1
    return line,

# ——— Run animation ———
ani = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)
plt.show()
