import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 10
k = 151 * (n - 2)  # spring constant, n-2 since we have a dashpot
l = 60e-3 / (n - 2)  # natural length of springs
mass = 1.8e-3 / (n - 2)  # mass of each small mass in chain
ds = 1e-3  # distance of needle to skin
df = ds + 1.55e-3  # adding thickness of skin to ds
dm = df + 10e-3  # adding thickness of fat to df
di = 16e-3  # distance from beginning of muscle to point of injection
mu_s = 1600  # stiffness of skin
mu_f = 500  # stiffness of fat
mu_m = 15000 # stiffness of muscle
spring_length = 0.035 / (n - 2)  # compressed length of springs

# plunger part
Mp = 0.0022  # mass of plunger
Ms = 0.0059  # mass of barrel

# Create mass matrix
m = np.diag([mass] * (n - 2) + [Mp] + [Ms])

eta = 10000  # viscocity coefficient
Nf = 1.72  # normal force of seal against syringe
mu_k = 0.7  # coefficient of friction for seal of syringe
C_p = 10  # coefficicient for viscous damping
A_p = 5.88e-5  # cross sectional area of syringe
volume_of_fluid = 0.3e-6  # volume that needs to be injected
mu = 0.00089 # fluid viscocity
Ln = 16e-3  # L = needle length
r = 0.205e-3  # needle radius
L = 34e-3
c = 0.01 # viscous damping

f_max = 150  # resistive force of skin before its punctured
u_puncture = 2.2e-3  # disaplacement of skin before it punctures
s = 1  # coefficient of tan (how sharp is transition)

F_friction = Nf * mu_k

A_DD = np.zeros((n, n))

for _ in range(n):
    if _ == 0:
        A_DD[_, _] = -k
        A_DD[_, _ + 1] = k
    elif _ == n - 2:
        A_DD[_, _] = -k
        A_DD[_, _ - 1] = k
    elif _ == n-1:
        continue # no spring connecting to plunger
    else:
        A_DD[_, _] = -2 * k
        A_DD[_, _ - 1] = k
        A_DD[_, _ + 1] = k

B_DD = np.zeros(n)
B_DD[0] = k * l

V_DD = np.zeros((n, n))

for _ in range(n):
    if _ == 0:
        V_DD[_, _] = -c
        V_DD[_, _ + 1] = c
    elif _ == n - 2:
        V_DD[_, _] = -c
        V_DD[_, _ - 1] = c
    elif _ == n-1:
        continue # no spring connecting to plunger
    else:
        V_DD[_, _] = -2 * c
        V_DD[_, _ - 1] = c
        V_DD[_, _ + 1] = c

dt = 0.00001
N = 1000000

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

total_volume = 0

Q_history = np.zeros(N)

# things to track

for i in range(1, N):
    if u[-1, i - 1] >= ds:
        break
    else:
        # Update B_DD for plunger/barrel
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
        B_DD[-1] = - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        # forces if there is no friction
        forces = A_DD @ u[:, i - 1] + B_DD + V_DD@v[:, i-1]

        # to check slip or stick condition
        if abs(forces[-2]) >= F_friction:
            v_rel = v[-2, i - 1] - v[-1, i - 1]  # if slipping this will be positive
            Q = max(v_rel * A_p, 0.0)
            total_volume += Q * dt
            Q_history[i - 1] = Q
            delta_p = 8 * mu * Ln * Q / (np.pi * r ** 4)
            if v_rel > 0:
                forces[-2] += -np.sign(forces[-2]) * F_friction - A_p * delta_p - C_p * v[-2, i - 1]
            else:
                forces[-2] += -np.sign(forces[-2]) * F_friction

            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        else:
            # they stick so move as a combined mass
            a[:, i] = np.linalg.solve(m, forces)
            a_combined = (forces[-2] + forces[-1]) / (m[-2, -2] + m[-1, -1])
            a[-2, i] = a_combined
            a[-1, i] = a_combined
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            # Fix first mass
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0
        if total_volume >= volume_of_fluid:
            break

end_phase1 = i-1

if total_volume < volume_of_fluid:
    # part where it enters skin
    for i in range(end_phase1, N):
        if u[-1, i - 1] >= df:
            break

        z = s * (u[-1, i - 1] - ds - u_puncture)
        A_DD[-1, -1] = - mu_s * np.tanh(z)
        B_DD[-1] = - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + ds * mu_s * np.tanh(z) + f_max * (1 - np.tanh(z))
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        # forces if there is no friction
        forces = A_DD @ u[:, i - 1] + B_DD  + V_DD@v[:, i-1]

        # to check slip or stick condition
        if abs(forces[-2]) >= F_friction:
            v_rel = v[-2, i - 1] - v[-1, i - 1]  # if slipping this will be positive
            Q = max(v_rel * A_p, 0.0)
            total_volume += Q * dt
            Q_history[i - 1] = Q
            delta_p = 8 * mu * Ln * Q / (np.pi * r ** 4)
            if v_rel > 0:
                forces[-2] += -np.sign(forces[-2]) * F_friction - A_p * delta_p - C_p * v[-2, i - 1]
            else:
                forces[-2] += -np.sign(forces[-2]) * F_friction

            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        else:
            # they stick so move as a combined mass
            a[:, i] = np.linalg.solve(m, forces)
            a_combined = (forces[-2] + forces[-1]) / (m[-2, -2] + m[-1, -1])
            a[-2, i] = a_combined
            a[-1, i] = a_combined
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            # Fix first mass
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        if total_volume >= volume_of_fluid:
            break


end_phase2 = i-1

if total_volume < volume_of_fluid:
    # moving onto fat through to muscle
    for i in range(end_phase2, N):
        if u[-1, i - 1] >= dm:
            break

        A_DD[-1, -1] = - mu_f
        B_DD[-1] = -A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) + mu_f * df
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        # forces if there is no friction
        forces = A_DD @ u[:, i - 1] + B_DD +  V_DD@v[:, i-1]

        # to check slip or stick condition
        if abs(forces[-2]) >= F_friction:
            v_rel = v[-2, i - 1] - v[-1, i - 1]  # if slipping this will be positive
            Q = max(v_rel * A_p, 0.0)
            total_volume += Q * dt
            Q_history[i - 1] = Q
            delta_p = 8 * mu * Ln * Q / (np.pi * r ** 4)
            if v_rel > 0:
                forces[-2] += -np.sign(forces[-2]) * F_friction - A_p * delta_p - C_p * v[-2, i - 1]
            else:
                forces[-2] += -np.sign(forces[-2]) * F_friction

            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        else:
            # they stick so move as a combined mass
            a[:, i] = np.linalg.solve(m, forces)
            a_combined = (forces[-2] + forces[-1]) / (m[-2, -2] + m[-1, -1])
            a[-2, i] = a_combined
            a[-1, i] = a_combined
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            # Fix first mass
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        if total_volume >= volume_of_fluid:
            break


end_phase3 = i-1

if total_volume < volume_of_fluid:
    # moving onto muscle through to point of injection
    for i in range(end_phase3, N):
        if u[-1, i - 1] >= di:
            break

        A_DD[-1, -1] = - mu_m
        B_DD[-1] = -A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) - mu_f * (dm - df) + mu_m * dm
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
        # forces if there is no friction
        forces = A_DD @ u[:, i - 1] + B_DD +  V_DD@v[:, i-1]

        # to check slip or stick condition
        if abs(forces[-2]) >= F_friction:
            v_rel = v[-2, i - 1] - v[-1, i - 1]  # if slipping this will be positive
            Q = max(v_rel * A_p, 0.0)
            total_volume += Q * dt
            Q_history[i - 1] = Q
            delta_p = 8 * mu * Ln * Q / (np.pi * r ** 4)
            if v_rel > 0:
                forces[-2] += -np.sign(forces[-2])*F_friction - A_p * delta_p - C_p * v[-2, i - 1]
            else:
                forces[-2] += -np.sign(forces[-2]) * F_friction


            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        else:
            # they stick so move as a combined mass
            a[:, i] = np.linalg.solve(m, forces)
            a_combined = (forces[-2] + forces[-1]) / (m[-2, -2] + m[-1, -1])
            a[-2, i] = a_combined
            a[-1, i] = a_combined
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]
            # Fix first mass
            a[0, i] = 0
            v[0, i] = 0
            u[0, i] = 0

        if total_volume >= volume_of_fluid:
            break


end_phase4 = i-1

for i in range(end_phase4, N):
    if total_volume >= volume_of_fluid:
        break
    else:

        Q = Q_history[i-1]
        delta_p = 8 * mu * Ln * Q / (np.pi * r**4)

        # Keep barrel stationary
        # Compute forces normally
        B_DD[-2] = A_p * eta / L * (v[-1, i-1] - v[-2, i-1]) - A_p*delta_p - C_p*v[-2,i-1] + k*l
        B_DD[-1] = 0
        A_DD[-1, -1] = 0


        forces = A_DD @ u[:, i-1] + B_DD +  V_DD@v[:, i-1]

        # Slip condition
        if abs(forces[-2]) >= F_friction:
            forces[-2] -= np.sign(forces[-2])*F_friction
            v_rel = v[-2, i-1] - v[-1, i-1]
            Q = max(v_rel * A_p, 0.0)
            total_volume += Q * dt
            Q_history[i-1] = Q

        forces_internal = A_DD[:-1, :-1] @ u[:-1, i - 1] + B_DD[:-1]
        a_internal = np.linalg.solve(m[:-1, :-1], forces_internal)
        v[:-1, i] = v[:-1, i - 1] + dt * a_internal
        u[:-1, i] = u[:-1, i - 1] + dt * v[:-1, i]

        # Barrel fixed
        a[-1, i] = 0
        v[-1, i] = 0
        u[-1, i] = u[-1, end_phase4-1]

        # First mass fixed
        a[0, i] = 0
        v[0, i] = 0
        u[0, i] = 0

end_phase5 = i-1

# plotting
print(f"All of the fluid injected in {dt * i} seconds")


u = u[:, 0:end_phase5]
v = v[:, 0:end_phase5]
a = a[:, 0:end_phase5]
Q_history = Q_history[0:end_phase5]

# NEW: Plot injection progress
plt.figure(figsize=(10, 6))
time_axis = np.arange(end_phase5) * dt
cumulative_volume = np.cumsum(np.where(Q_history > 0, Q_history * dt, 0))
plt.plot(time_axis, cumulative_volume)
plt.axhline(y=volume_of_fluid, color='r', linestyle='--', label='Target Volume')
plt.xlabel('Time (s)')
plt.ylabel('Injected Volume (m³)')
plt.title('Fluid Injection Progress')
plt.legend()
plt.grid(True)
plt.show()

# Rest of your plotting code remains the same...
# animating
total_steps = i
fig, ax = plt.subplots()
line, = ax.plot(np.arange(n), u[:, 0], 'o-', lw=2)
ax.set_xlim(0, n - 1)
ax.set_ylim(0, di+1) # maaasive? low taper fade, do we need plus 1m seems mmmmmmmmmmmmmmmmmmassive
ax.set_xlabel("Mass Index")
ax.set_ylabel("Displacement")
ax.set_title("Displacement of Masses Over Time")


def update(frame):
    line.set_ydata(u[:, frame])
    ax.set_title(f"Time: {frame * dt:.2f}s")
    return line,


ani = animation.FuncAnimation(fig, update, frames=range(0, total_steps, 2),
                              interval=30, blit=True)

plt.show()

plt.plot([i for i in range(end_phase5)], u[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], u[-2, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of plunger")
plt.show()

plt.plot([i for i in range(end_phase5)], v[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], v[-2, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of plunger")
plt.show()

plt.plot([i for i in range(end_phase5)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], Ms * a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], Mp * a[-2, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of plunger")
plt.show()