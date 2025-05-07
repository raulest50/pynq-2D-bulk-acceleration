import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the parameters of the wave
length = 10  # Length of the spatial domain
time = 5  # Total simulation time
dt = 0.01  # Time step
dx = 0.1  # Spatial step
c = 1.0  # Wave speed

# Create the spatial and temporal grids
x = np.arange(0, length, dx)
t = np.arange(0, time, dt)
n_t = len(t)

# Initialize the wave amplitude
u = np.zeros_like(x)

# Set initial condition (e.g., a Gaussian pulse)
center = length / 2
sigma = 1
u = np.exp(-0.5 * ((x - center) / sigma) ** 2)

# Create the figure and axes for the animation
fig, ax = plt.subplots()
line, = ax.plot(x, u)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Wave Propagation")


# Define the animation function
def animate(i):
    global u
    # Numerical solution using finite differences (example: simple advection)
    u_new = np.zeros_like(u)
    for j in range(1, len(x)):
        u_new[j] = u[j] - c * dt / dx * (u[j] - u[j - 1])
    u_new[0] = u_new[-1]  # Periodic boundary condition
    u = u_new

    line.set_ydata(u)
    return line,


# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=n_t, interval=20, blit=True)

# Display the animation
plt.show()