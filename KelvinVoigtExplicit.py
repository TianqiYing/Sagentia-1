import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n = 10
k =  1500*(n-2) # spring constant, n-2 since we have a dashpot
l = 60e-3 / (n-1)  # natural length of springs
mass = 1.8e-3 / n-1 # mass of each small mass in chain
ds = 1e-3 # distance of needle to skin
df = ds + 1.55e-3 # adding thickness of skin to ds
dm = df + 10e-3 # adding thickness of fat to df
di = 16e-3 # distance from beginning of muscle to point of injection
mu_s = 150 # stiffness of skin
mu_f = 20 # stiffness of fat
mu_m = 300 # stiffness of muscle
spring_length = 0.035 / (n-2) # compressed length of springs


# plunger part

Mp = 0.0012 # mass of plunger
Ms = 0.0022 # mass of barrel

# Create mass matrix
m = np.diag([mass]*(n-2) + [Mp] + [Ms])

eta = 1e-3
L = 0.016
Nf = 2 # normal force of seal against syringe
mu_k = 8 # coefficient of friction for seal of syringe
C_p = 0.033 # coefficicient for viscous damping
A_p = 5.88e-5 # cross sectional area of syringe
volume_of_fluid = 0.3e-3 # volume that needs to be injected
mu = 0.00089 # fluid viscocity
L = 16e-3 # L = needle length
r = 0.135e-3 # needle radius

F_friction = Nf*mu_k

A_DD = np.zeros((n, n))

for _ in range(n):
    if _ == 0:
        A_DD[_, _] = -k
        A_DD[_, _+1] = k
    elif _ == n-2:
        A_DD[_, _] = -k
        A_DD[_, _-1] = k
    elif _ == n-1:
        continue # all zeros since nothing depends on displacement yet
    else:
        A_DD[_, _] = -2*k
        A_DD[_, _-1] = k
        A_DD[_, _+1] = k

B_DD = np.zeros(n)
B_DD[0] = -k*l

dt = 0.000001
N = 10000

initial_positions = np.linspace(0, spring_length, n)
natural_positions = np.linspace(0, l, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))


# things to track

for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1] < ds:
        # changing B_DD after each iteration
        B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) # terms for plunger
        B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        a[:, i] = (A_DD @ u[:, i - 1] + B_DD) / np.diag(m)
        v[:, i] = v[:, i - 1] + dt * a[:, i]  # ADD THIS LINE
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        F_plunger = B_DD[-2]  # or the total force acting on the plunger mass

        if abs(F_plunger) < F_friction:
            # barrel moves with plunger
            a[-1, i] = a[-2, i]
        else:
            # barrel slips relative to plunger
            a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

        v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
        u[-1, i] = u[-1, i - 1] + dt * v[-1, i]

        # ensuring first mass stays still
        v[0, i] = 0
        u[0, i] = 0


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
        B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z))
        B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) # terms for plunger

        a[:, i] = (A_DD @ u[:, i - 1] + B_DD) / np.diag(m)
        v[:, i] = v[:, i - 1] + dt * a[:, i]  # ADD THIS LINE
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        F_plunger = B_DD[-2]  # or the total force acting on the plunger mass

        if abs(F_plunger) < F_friction:
            # barrel moves with plunger
            a[-1, i] = a[-2, i]
        else:
            # barrel slips relative to plunger
            a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

        v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
        u[-1, i] = u[-1, i - 1] + dt * v[-1, i]

        # ensuring first mass stays still
        v[0, i] = 0
        u[0, i] = 0
    else:
        break

end_phase2 = i


# moving onto fat through to muscle





for i in range(end_phase2, N):
    if u[-1, i-1] < dm:
        A_DD[-1, -1] = -k - mu_f
        B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) + mu_f * df
        B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) # terms for plunger

        a[:, i] = (A_DD @ u[:, i - 1] + B_DD) / np.diag(m)
        v[:, i] = v[:, i - 1] + dt * a[:, i]  # ADD THIS LINE
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        F_plunger = B_DD[-2]  # or the total force acting on the plunger mass

        if abs(F_plunger) < F_friction:
            # barrel moves with plunger
            a[-1, i] = a[-2, i]
        else:
            # barrel slips relative to plunger
            a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

        v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
        u[-1, i] = u[-1, i - 1] + dt * v[-1, i]

        # ensuring first mass stays still
        v[0, i] = 0
        u[0, i] = 0

    else:
        break

end_phase3 = i


# moving onto muscle through to point of injection


for i in range(end_phase3, N):
    if u[-1, i-1] < di:
        B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) - mu_f * (dm - df) + mu_m * dm
        B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) # terms for plunger

        A_DD[-1, -1] = -k - mu_m

        a[:, i] = (A_DD @ u[:, i - 1] + B_DD) / np.diag(m)
        v[:, i] = v[:, i - 1] + dt * a[:, i]  # ADD THIS LINE
        u[:, i] = u[:, i - 1] + dt * v[:, i]

        F_plunger = B_DD[-2]  # or the total force acting on the plunger mass

        if abs(F_plunger) < F_friction:
            # barrel moves with plunger
            a[-1, i] = a[-2, i]
        else:
            # barrel slips relative to plunger
            a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

        v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
        u[-1, i] = u[-1, i - 1] + dt * v[-1, i]

        # ensuring first mass stays still
        v[0, i] = 0
        u[0, i] = 0
    else:
        break

end_phase4 = i

# plunger part

Q = A_p*v[-1, i - 1]
total_volume = 0

for i in range(end_phase4, N):
    if total_volume < volume_of_fluid:
        if Q > 0:
            total_volume += Q*dt

        delta_p = 8 * mu * L * Q / np.pi * r**4

        # Update force vector for internal masses only
        B_DD_internal = B_DD.copy()
        B_DD_internal[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + Nf*mu_k + A_p*delta_p + C_p*v[-1, i - 1]
        B_DD_internal[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

        # Solve for internal masses (excluding last mass)
        a_internal = (A_DD[:-1, :-1] @ u[:-1, i - 1] + B_DD_internal[:-1]) / np.diag(m[:-1])
        v[:-1, i] = v[:-1, i - 1] + dt * a_internal
        u[:-1, i] = u[:-1, i - 1] + dt * v[:-1, i]
        a[:-1, i] = a_internal

        # Force on the plunger mass (last mass)
        F_plunger = B_DD_internal[-2]

        if abs(F_plunger) < F_friction:
            # barrel moves with plunger
            a[-1, i] = a[-2, i]
        else:
            # barrel slips relative to plunger
            a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

        v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
        u[-1, i] = u[-1, i - 1] + dt * v[-1, i]

        # First mass stays fixed
        v[0, i] = 0
        u[0, i] = 0

        # Update Q for next iteration
        Q = A_p * v[-1, i - 1]

        print(f"{total_volume}, {volume_of_fluid}, {Q}")

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