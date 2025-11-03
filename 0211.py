import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 10
k = 1500 * (n - 3)  # spring constant, n-2 since we have a dashpot
l = 60e-3 / (n - 3)  # natural length of springs
mass = 1.8e-3 / (n - 2)  # mass of each small mass in chain
ds = 1e-3  # distance of needle to skin
df = ds + 1.55e-3  # adding thickness of skin to ds
dm = df + 10e-3  # adding thickness of fat to df
di = 16e-3  # distance from beginning of muscle to point of injection
mu_s = 1000  # stiffness of skin
mu_f = 20000  # stiffness of fat
mu_m = 100000  # stiffness of muscle
spring_length = 0.035 / (n - 3)  # compressed length of springs

# plunger part
Mp = 0.002  # mass of plunger
Ms = 0.0035  # mass of barrel

# Create mass matrix
m = np.diag([mass] * (n - 2) + [Mp] + [Ms])

eta = 10  # viscocity coefficient
Nf = 2  # normal force of seal against syringe
mu_k = 0.5  # coefficient of friction for seal of syringe
C_p = 0.033  # coefficicient for viscous damping
A_p = 5.88e-5  # cross sectional area of syringe
volume_of_fluid = 0.3e-6  # volume that needs to be injected
mu = 0.00089  # fluid viscocity
Ln = 16e-3  # L = needle length
r = 0.205e-3  # needle radius
L = 34e-3

f_max = 150000  # resistive force of skin before its punctured
u_puncture = 2.2e-3  # disaplacement of skin before it punctures
s = 1000  # coefficient of tan (how sharp is transition)

F_friction = Nf * mu_k

A_DD = np.zeros((n, n))

for _ in range(n):
    if _ == 0:
        A_DD[_, _] = -k
        A_DD[_, _ + 1] = k
    elif _ == n - 2:
        A_DD[_, _] = -k
        A_DD[_, _ - 1] = k
    elif _ == n - 1:
        A_DD[_, _] = 0  # all zeros since nothing depends on displacement yet
    else:
        A_DD[_, _] = -2 * k
        A_DD[_, _ - 1] = k
        A_DD[_, _ + 1] = k

B_DD = np.zeros(n)
B_DD[0] = -k * l

dt = 0.0001
N = 100000

initial_positions = np.linspace(0, spring_length, n)
natural_positions = np.linspace(0, l, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

# NEW: Track fluid injection from the beginning
total_volume = 0
injection_started = False
Q_history = np.zeros(N)

# things to track

for i in range(1, N):
    if u[-1, i - 1] >= ds:
        break
    else:
        # Update B_DD for plunger/barrel
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
        B_DD[-1] = - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        # Compute forces if barrel sticks
        forces_if_stuck = A_DD @ u[:, i - 1] + B_DD
        F_plunger_if_stuck = forces_if_stuck[-2]

        # CHANGED: compute slip gating and relative plunger-barrel velocity
        slip = abs(F_plunger_if_stuck) >= F_friction  # CHANGED
        v_rel = v[-2, i - 1] - v[-1, i - 1]           # CHANGED
        Q = (A_p * max(v_rel, 0.0)) if slip else 0.0  # CHANGED

        total_volume += Q * dt  # CHANGED: gated implicitly by slip
        Q_history[i] = Q

        # Adjust for slip if necessary
        forces = forces_if_stuck.copy()
        if abs(F_plunger_if_stuck) >= F_friction:
            forces[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck) * F_friction

        # Solve once
        a[:, i] = np.linalg.solve(m, forces)
        v[:, i] = v[:, i - 1] + dt * a[:, i]
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        # Fix first mass
        v[0, i] = 0
        u[0, i] = 0

        # NEW: Check if we've injected enough fluid
        if total_volume >= volume_of_fluid:
            break

end_phase1 = i

if total_volume < volume_of_fluid:
    # part where it enters skin
    for i in range(end_phase1, N):
        if u[-1, i - 1] >= df:
            break

        z = s * (u[-1, i - 1] - ds - u_puncture)
        A_DD[-1, -1] = -k - mu_s * np.tanh(z)
        B_DD[-1] = - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + ds * mu_s * np.tanh(z) + f_max * (1 - np.tanh(z))
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        forces_if_stuck = A_DD @ u[:, i - 1] + B_DD
        F_plunger_if_stuck = forces_if_stuck[-2]

        # CHANGED: compute slip gating and relative plunger-barrel velocity
        slip = abs(F_plunger_if_stuck) >= F_friction  # CHANGED
        v_rel = v[-2, i - 1] - v[-1, i - 1]           # CHANGED
        Q = (A_p * max(v_rel, 0.0)) if slip else 0.0  # CHANGED

        total_volume += Q * dt  # CHANGED: gated implicitly by slip
        Q_history[i] = Q

        forces = forces_if_stuck.copy()
        if abs(F_plunger_if_stuck) >= F_friction:
            forces[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck) * F_friction

        a[:, i] = np.linalg.solve(m, forces)
        v[:, i] = v[:, i - 1] + dt * a[:, i]
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        # fixing first mass
        v[0, i] = 0
        u[0, i] = 0

        # NEW: Check if we've injected enough fluid
        if total_volume >= volume_of_fluid:
            break

end_phase2 = i

if total_volume < volume_of_fluid:
    # moving onto fat through to muscle
    for i in range(end_phase2, N):
        if u[-1, i - 1] >= dm:
            break

        A_DD[-1, -1] = -k - mu_f
        B_DD[-1] = -A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) + mu_f * df
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        forces_if_stuck = A_DD @ u[:, i - 1] + B_DD
        F_plunger_if_stuck = forces_if_stuck[-2]

        # CHANGED: compute slip gating and relative plunger-barrel velocity
        slip = abs(F_plunger_if_stuck) >= F_friction  # CHANGED
        v_rel = v[-2, i - 1] - v[-1, i - 1]  # CHANGED
        Q = (A_p * max(v_rel, 0.0)) if slip else 0.0  # CHANGED

        total_volume += Q * dt  # CHANGED: gated implicitly by slip
        Q_history[i] = Q

        forces = forces_if_stuck.copy()
        if abs(F_plunger_if_stuck) >= F_friction:
            forces[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck) * F_friction

        a[:, i] = np.linalg.solve(m, forces)
        v[:, i] = v[:, i - 1] + dt * a[:, i]
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        v[0, i] = 0 # still needed in this phase ?
        u[0, i] = 0  # still needed in this phase ?

        # NEW: Check if we've injected enough fluid
        if total_volume >= volume_of_fluid:
            break

end_phase3 = i

if total_volume < volume_of_fluid:
    # moving onto muscle through to point of injection
    for i in range(end_phase3, N):
        if u[-1, i - 1] >= di:
            break

        A_DD[-1, -1] = -k - mu_m
        B_DD[-1] = -A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) - mu_f * (dm - df) + mu_m * dm
        B_DD[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        forces_if_stuck = A_DD @ u[:, i - 1] + B_DD
        F_plunger_if_stuck = forces_if_stuck[-2]

        # CHANGED: compute slip gating and relative plunger-barrel velocity
        slip = abs(F_plunger_if_stuck) >= F_friction  # CHANGED
        v_rel = v[-2, i - 1] - v[-1, i - 1]  # CHANGED
        Q = (A_p * max(v_rel, 0.0)) if slip else 0.0  # CHANGED

        total_volume += Q * dt  # CHANGED: gated implicitly by slip
        Q_history[i] = Q

        forces = forces_if_stuck.copy()
        if abs(F_plunger_if_stuck) >= F_friction:
            forces[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck) * F_friction

        a[:, i] = np.linalg.solve(m, forces)
        v[:, i] = v[:, i - 1] + dt * a[:, i]
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        v[0, i] = 0  # still needed ?
        u[0, i] = 0  # still needed ?

        # NEW: Check if we've injected enough fluid
        if total_volume >= volume_of_fluid:
            break

end_phase4 = i

if total_volume < volume_of_fluid:
    # plunger part - only if we haven't injected all fluid yet
    for i in range(end_phase4, N):
        # CHANGED: compute slip gating and relative plunger-barrel velocity
        # Compute flow rate from plunger using relative motion, gated by slip
        # We need F_plunger_if_stuck here as well, recompute with internal forces below
        Q = 0.0  # placeholder, will be updated after forces are built

        if total_volume >= volume_of_fluid:
            break

        # CHANGED: after we compute forces, recompute slip and Q
        # Pressure drop in the needle will use this Q

        # Update B_DD including all resistances for the barrel
        B_DD_internal = B_DD.copy()
        B_DD_internal[-1] = (
                A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
                + Nf * mu_k
                + A_p * delta_p
                + C_p * v[-1, i - 1]
        )
        B_DD_internal[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        forces_internal = A_DD[:-1, :-1] @ u[:-1, i - 1] + B_DD_internal[:-1]
        F_plunger_if_stuck = forces_internal[-1]

        # CHANGED: compute slip gating and relative plunger-barrel velocity (late-phase)
        slip = abs(F_plunger_if_stuck) >= F_friction  # CHANGED
        v_rel = v[-2, i - 1] - v[-1, i - 1]           # CHANGED
        Q = (A_p * max(v_rel, 0.0)) if slip else 0.0  # CHANGED

        # Pressure drop in the needle
        delta_p = 8 * mu * Ln * Q / (np.pi * r ** 4)

        # Update B_DD including all resistances for the barrel
        B_DD_internal = B_DD.copy()
        B_DD_internal[-1] = (
                A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
                + Nf * mu_k
                + A_p * delta_p
                + C_p * v[-1, i - 1]
        )
        B_DD_internal[-2] = k * l + A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        forces_internal = A_DD[:-1, :-1] @ u[:-1, i - 1] + B_DD_internal[:-1]
        F_plunger_if_stuck = forces_internal[-1]

        forces_internal_full = np.zeros(n)
        forces_internal_full[:-1] = forces_internal
        forces_internal_full[-1] = F_plunger_if_stuck
        if abs(F_plunger_if_stuck) >= F_friction:
            forces_internal_full[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck) * F_friction

        a_internal = np.linalg.solve(m, forces_internal_full)
        v[:, i] = v[:, i - 1] + dt * a_internal
        u[:, i] = u[:, i - 1] + dt * v[:, i]
        a[:, i] = a_internal

        v[0, i] = 0
        u[0, i] = 0

        v[-1, i] = v[-1, i - 1]
        u[-1, i] = u[-1, i - 1]

        print(f"Phase 5: {total_volume:.6f}, {volume_of_fluid:.6f}, {Q:.6f}")

end_phase5 = i

# plotting
print(f"All of the fluid injected in {dt * i} seconds")
print(f"Final injected volume: {total_volume:.6f} m³")
print(f"Target volume: {volume_of_fluid:.6f} m³")

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
ax.set_ylim(0, di + 1)
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

plt.plot([i for i in range(end_phase5)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase5)], Ms * a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()