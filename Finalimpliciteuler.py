import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# for simple 3 mass system
n = 10
k =  1500 # was 10
l = 60e-3  # was 1
m = 1.8e-3 # was 0.5
M = 0.02 # was 5 need to make this more accurate but its ok for now
ds = 2e-3 # was 2
df = ds + 2.5e-3 # was 3
dm = df + 10e-3 # was 4 this is waff
di = 16e-3 # was 4.5
mu_s = 150 # this was 1, need to reference that another time
mu_f = 20 # this was 0.5 need to reference
mu_m = 300 # was 2, need to reference this
spring_length = 0.035 # was 1 complete waff

# plunger part
Nf = 1
mu_k = 1
C_p = 1
A_p = 1
delta_p = 1
volume_of_fluid = 1


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

dt = 0.001
N = 10000
initial_positions = np.linspace(0, spring_length, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

u[:, 0] = initial_positions

# Create mass matrix
mass_matrix = np.diag([m]*(n-1) + [M])

# things to track


for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1] < ds:
        # Implicit Euler: (I - dt^2 * M^-1 * A) * u[i] = u[i-1] + dt*v[i-1]
        LHS = np.eye(n) - dt**2 * np.linalg.inv(mass_matrix) @ A_DD
        RHS = u[:, i-1] + np.linalg.inv(mass_matrix)@dt*v[:, i-1] + dt**2 * np.linalg.inv(mass_matrix) @ B_DD
        u[:, i] = np.linalg.solve(LHS, RHS)
        u[0, i] = 0
        v[:, i] = (u[:, i] - u[:, i-1])/dt
        a[:, i] = (v[:, i] - v[:, i-1])/dt
    else:
        break

end_phase1 = i


# part where it enters skin

f_max = 3
u_puncture = 2.2
s = 0.5

for i in range(end_phase1, N):
    if u[-1, i-1] < df:
        z = s*(u[-1, i-1] - ds - u_puncture)
        A_DD[-1, -1] = -k - mu_s*np.tanh(z)
        B_DD[-1] = k * l + ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z))

        LHS = np.eye(n) - dt**2 * np.linalg.inv(mass_matrix) @ A_DD
        RHS = u[:, i-1] + dt*v[:, i-1] + dt**2 * np.linalg.inv(mass_matrix) @ B_DD
        u[:, i] = np.linalg.solve(LHS, RHS)
        u[0, i] = 0
        v[:, i] = (u[:, i] - u[:, i-1])/dt
        a[:, i] = (v[:, i] - v[:, i-1])/dt
    else:
        break

end_phase2 = i


# moving onto fat through to muscle


A_DD[-1, -1] = -k - mu_f
B_DD[-1] = k*l - mu_s*(df - ds) + mu_f*df

for i in range(end_phase2, N):
    if u[-1, i-1] < dm:
        LHS = np.eye(n) - dt**2 * np.linalg.inv(mass_matrix) @ A_DD
        RHS = u[:, i-1] + dt*v[:, i-1] + dt**2 * np.linalg.inv(mass_matrix) @ B_DD
        u[:, i] = np.linalg.solve(LHS, RHS)
        u[0, i] = 0
        v[:, i] = (u[:, i] - u[:, i-1])/dt
        a[:, i] = (v[:, i] - v[:, i-1])/dt
    else:
        break

end_phase3 = i


# moving onto muscle through to point of injection

A_DD[-1, -1] = -k - mu_m
B_DD[-1] = k*l - mu_s*(df - ds) - mu_f*(dm - df) + mu_m*dm

for i in range(end_phase3, N):
    if u[-1, i-1] < di:
        LHS = np.eye(n) - dt**2 * np.linalg.inv(mass_matrix) @ A_DD
        RHS = u[:, i-1] + dt*v[:, i-1] + dt**2 * np.linalg.inv(mass_matrix) @ B_DD
        u[:, i] = np.linalg.solve(LHS, RHS)
        u[0, i] = 0
        v[:, i] = (u[:, i] - u[:, i-1])/dt
        a[:, i] = (v[:, i] - v[:, i-1])/dt
    else:
        break

end_phase4 = i

# plunger part

Q = A_p*v[-1, i - 1]

mu = 1 # fluid viscocity
L = 1 # L = needle length
r = 1 # needle radius

for i in range(end_phase4, N): # keep track of i for future for loops
    if Q*i*dt < volume_of_fluid: # change condition to something abt v ? ?
        delta_p = 8*mu*L*Q / np.pi*r**4
        B_DD[-1] = k*l + Nf*mu_k + A_p*delta_p + C_p*v[-1, i - 1]
        LHS = np.eye(n) - dt**2 * np.linalg.inv(mass_matrix) @ A_DD
        RHS = u[:, i-1] + dt*v[:, i-1] + dt**2 * np.linalg.inv(mass_matrix) @ B_DD
        u[:, i] = np.linalg.solve(LHS, RHS)
        u[0, i] = 0
        v[:, i] = (u[:, i] - u[:, i-1])/dt
        a[:, i] = (v[:, i] - v[:, i-1])/dt
        Q = A_p * v[-1, i - 1]
    else:
        break

end_phase5 = i



# plotting

u = u[:, 0:end_phase5]
v = v[:, 0:end_phase5]
a = a[:, 0:end_phase5]

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

plt.plot([i for i in range(end_phase5)], u[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], v[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], M*a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()


