import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# for simple 3 mass system
n = 10
k =  10
l = 1
m = 1
M = 5
ds = 2
df = 3
dm = 4
di = 4.5
mu_s = 1
mu_f = 0.5
mu_m = 2
spring_length = 1
A_DD = np.zeros((n, n))

for _ in range(n):
    if _ == 0:
        A_DD[_, _] = -k
        A_DD[_, _+1] = k
    elif _ == n-1:
        A_DD[_, _] = -k
        A_DD[_, _-1] = k
    else:
        A_DD[_, _] = -2*k
        A_DD[_, _-1] = k
        A_DD[_, _+1] = k

B_DD = np.zeros(n)
B_DD[0] = -k*l
B_DD[-1] = k*l

dt = 0.01
N = 10000
initial_positions = np.linspace(0, spring_length, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

u[:, 0] = initial_positions

# things to track


for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1] < ds:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
    else:
        break

end_phase1 = i


# part where it enters skin

A_DD[-1, -1] = -k - mu_s
B_DD[-1] = k*l + mu_s*ds

for i in range(end_phase1, N):
    if u[-1, i-1] < df:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
    else:
        break

end_phase2 = i


# moving onto fat through to muscle

A_DD[-1, -1] = -k - mu_f
B_DD[-1] = k*l - mu_s*(df - ds) + mu_f*df

for i in range(end_phase2, N):
    if u[-1, i-1] < dm:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
    else:
        break

end_phase2 = i


# moving onto muscle through to point of injection

A_DD[-1, -1] = -k - mu_m
B_DD[-1] = k*l - mu_s*(df - ds) - mu_f*(dm - df) + mu_m*dm

for i in range(end_phase2, N):
    if u[-1, i-1] < di:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
    else:
        break

end_phase3 = i
u = u[:, 0:end_phase3]
v = v[:, 0:end_phase3]
a = a[:, 0:end_phase3]

# animating
total_steps = i
fig, ax = plt.subplots()
line, = ax.plot(np.arange(n), u[:, 0], 'o-', lw=2)
ax.set_xlim(0, n-1)
ax.set_ylim(0, di + 1)
ax.set_xlabel("Mass Index")
ax.set_ylabel("Displacement")
ax.set_title("Displacement of Masses Over Time")

def update(frame):
    line.set_ydata(u[:, frame])
    ax.set_title(f"Time: {frame*dt:.2f}s")
    return line,

ani = animation.FuncAnimation(fig, update, frames=range(0, total_steps, 2),
                              interval=30, blit=True)

plt.show()

plt.plot([i for i in range(end_phase3)], u[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of syringe")
plt.show()

print(len(u[-1, :]), len(v[-1, :]))

plt.plot([i for i in range(end_phase3)], v[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of syringe")
plt.show()

plt.plot([i for i in range(end_phase3)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase3)], M*a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()


# Summarising ODE system
#
# Modelled the spring as a series of spring-mass chains. This is beneficial since it allows the modelling of wave propagation through the spring which will be a factor in the case of the epipen. With a simple model of one spring and one mass we are unable to see what happens within the spring.
#
# The lagrangian was used to create ODE’s for the motion of each mass in the chain. Where potential energy is dependant on the extension or compression of springs either side of the mass and the kinetic energy was calculated using the second derivative of displacement wrt t. Important to note that the final mass in the chain is the mass of the syringe. The rest of the masses are the mass of the spring divided by the number of spring-mass chains distributed.
#
# This creates a system of ODE’s where each ODE describes the displacement of a mass from its initial position.
#
# To model the motion of the spring through the persons leg, three constants were used to consider the friction against the needle whilst it goes through the skin, fat and muscle. The system of ODE’s now has four stages (loaded to touching leg, going through skin, going through fat and going through muscle). The resistance from the skin, fat or muscle was modelled as proportional to the distance the needle is through the layer since the needle is now in contact with more of this material.



