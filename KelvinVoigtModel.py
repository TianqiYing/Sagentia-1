import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# for simple 3 mass system
n = 10
k =  300*(n-1) # was 10
l = 60e-3 / (n-1)  # was 1
m = 1.8e-3 / n # was 0.5
ds = 2e-3 # was 2
df = ds + 2.5e-3 # was 3
dm = df + 10e-3 # was 4 this is waff
di = 16e-3 # was 4.5
mu_s = 150 # this was 1, need to reference that another time
mu_f = 20 # this was 0.5 need to reference
mu_m = 300 # was 2, need to reference this
spring_length = 0.035 # was 1 complete waff

# plunger part
Nf = 2 # normal force of seal against syringe -- need to add 'break free' force
mu_k = 8 # coefficient of friction for seal of syringe
C_p = 0.033 # coefficicient for viscous damping
A_p = 5.88e-5 # cross sectional area of syringe
volume_of_fluid = 0.3e-3


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

dt = 0.0001
N = 10000
initial_positions = np.linspace(0, spring_length, n)
natural_positions = np.linspace(0, l, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

Mp = 0.0012 # mass of plunger
Ms = 0.0022 # mass of barrel
# Create mass matrix
m = np.diag([m]*(n-2) + [Mp] + [Ms])
eta = 1e-3
L = 0.016


# things to track
for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1]  < ds:
        # Implicit Euler: (I - dt^2 * M^-1 * A) * u[i] = u[i-1] + dt*v[i-1]
        LHS = m / dt - (A_p * eta / L) * np.eye(n) - A_DD * dt
        RHS = (m / dt + (A_p * eta / L) * np.eye(n)) @ v[:, i - 1] + A_DD @ u[:, i - 1] + B_DD
        v[:, i] = np.linalg.solve(LHS, RHS)
        u[:, i] = v[:, i] * dt + u[:, i - 1]
        u[0, i] = 0
        v[0, i] = 0
        a[:, i] = (v[:, i] - v[:, i - 1]) / dt

        # first one stays fixed
        u[0, i] = 0
        v[0, i] = 0.0
    else:
        break

end_phase1 = i


# part where it enters skin

f_max = 10
u_puncture = 1e-3
s = 0.5

for i in range(end_phase1, N):
    if u[-1, i-1] < df:
        z = s*(u[-1, i-1] - ds - u_puncture)
        A_DD[-1, -1] = -k - mu_s*np.tanh(z)
        B_DD[-1] = k * l + ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z))

        LHS = m / dt - (A_p * eta / L) * np.eye(n) - A_DD * dt
        RHS = (m / dt + (A_p * eta / L) * np.eye(n)) @ v[:, i - 1] + A_DD @ u[:, i - 1] + B_DD
        v[:, i] = np.linalg.solve(LHS, RHS)
        u[:, i] = v[:, i] * dt + u[:, i - 1]
        u[0, i] = 0
        a[:, i] = (v[:, i] - v[:, i - 1]) / dt

        # first one stays fixed
        u[0, i] = 0
        v[0, i] = 0.0
    else:
        break

end_phase2 = i


# moving onto fat through to muscle


A_DD[-1, -1] = -k - mu_f
B_DD[-1] = k*l - mu_s*(df - ds) + mu_f*df

for i in range(end_phase2, N):
    if u[-1, i-1] < dm:
        LHS = m / dt - (A_p * eta / L) * np.eye(n) - A_DD * dt
        RHS = (m / dt + (A_p * eta / L) * np.eye(n)) @ v[:, i - 1] + A_DD @ u[:, i - 1] + B_DD
        v[:, i] = np.linalg.solve(LHS, RHS)
        u[:, i] = v[:, i] * dt + u[:, i - 1]
        u[0, i] = 0
        a[:, i] = (v[:, i] - v[:, i - 1]) / dt

        # first one stays fixed
        u[0, i] = 0
        v[0, i] = 0.0
    else:
        break

end_phase3 = i


# moving onto muscle through to point of injection

A_DD[-1, -1] = -k - mu_m
B_DD[-1] = k*l - mu_s*(df - ds) - mu_f*(dm - df) + mu_m*dm

for i in range(end_phase3, N):
    if u[-1, i-1]  < di:
        LHS = m / dt - (A_p * eta / L) * np.eye(n) - A_DD * dt
        RHS = (m / dt + (A_p * eta / L) * np.eye(n)) @ v[:, i - 1] + A_DD @ u[:, i - 1] + B_DD
        v[:, i] = np.linalg.solve(LHS, RHS)
        u[:, i] = v[:, i]*dt + u[:, i-1]
        u[0, i] = 0
        a[:, i] = (v[:, i] - v[:, i - 1]) / dt

        # first one stays fixed
        u[0, i] = 0
        v[0, i] = 0.0
    else:
        break

end_phase4 = i

# plunger part

A_DD[-1, -1] = -k
Q = A_p*v[-2, i - 1]

mu = 0.00089 # fluid viscocity

L = 16e-3 # L = needle length
r = 0.135e-3 # needle radius
# initial conditions for plunger part
total_volume = 0

for i in range(end_phase4, N): # keep track of i for future for loops
    if  total_volume < volume_of_fluid:
        if Q > 0:
            total_volume += Q*dt
        else:
            total_volume += 0
        print(f"{total_volume}, {volume_of_fluid}, {Q}")

        delta_p = 8*mu*L*Q / np.pi*r**4
        B_DD[-1] = k*l + Nf*mu_k + A_p*delta_p + C_p*v[-1, i - 1]
        LHS = m / dt - (A_p * eta / L) * np.eye(n) - A_DD * dt
        RHS = (m / dt + (A_p * eta / L) * np.eye(n)) @ v[:, i - 1] + A_DD @ u[:, i - 1] + B_DD

        # only solving for 1 to n-1 since first and last fixed
        free_index = np.arange(n-1)
        LHS_red = LHS[np.ix_(free_index, free_index)]
        RHS_red = RHS[free_index]

        v_free = np.linalg.solve(LHS_red, RHS_red)
        v[free_index, i] = v_free
        v[-1, i] = 0.0
        v[0, i] = 0.0

        u[:, i] = u[:, i-1] + dt*v[:, i]
        a[:, i] = (v[:, i] - v[:, i-1] )/ dt

        # first one stays fixed
        u[0, i] = 0
        Q = A_p * v[-2, i - 1]

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

plt.plot([i for i in range(end_phase5)], Ms*a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()