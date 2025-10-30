import numpy as np
import matplotlib.pyplot as plt

# --- Constants (same as before) ---
n = 10
l = 60e-3 / (n-2)
mass = 1.8e-3 / (n-1)
ds = 1e-3
df = ds + 1.55e-3
dm = df + 10e-3
di = 16e-3
mu_s = 5000
mu_f = 20000
mu_m = 100000
spring_length = 0.035 / (n-2)

Mp = 0.0012
Ms = 0.0022
m = np.diag([mass]*(n-2) + [Mp] + [Ms])

eta = 1e-3
Nf = 2
mu_k = 8
C_p = 0.033
A_p = 5.88e-5
volume_of_fluid_target = 0.3e-3
mu = 0.00089
r = 0.135e-3
L = 34e-3

f_max = 150000
u_puncture = 2.2e-3
s = 0.5

F_friction = Nf*mu_k

dt = 0.0001
N = 10000

# --- Parameter ranges ---
k_values = np.linspace(1000*(n-2), 5000*(n-2), 5)   # example range of spring constants
Ln_values = np.linspace(10e-3, 20e-3, 5)            # example range of needle lengths

# --- Result matrix ---
fluid_injected = np.zeros((len(k_values), len(Ln_values)))

# --- Simulation function ---
def simulate_injection(k, Ln):
    # Initialize matrices
    A_DD = np.zeros((n, n))
    for _ in range(n):
        if _ == 0:
            A_DD[_, _] = -k
            A_DD[_, _+1] = k
        elif _ == n-2:
            A_DD[_, _] = -k
            A_DD[_, _-1] = k
        elif _ == n-1:
            A_DD[_, _] = 0
        else:
            A_DD[_, _] = -2*k
            A_DD[_, _-1] = k
            A_DD[_, _+1] = k

    B_DD = np.zeros(n)
    B_DD[0] = -k*l

    u = np.zeros((n, N))
    v = np.zeros((n, N))
    a = np.zeros((n, N))

    total_volume = 0
    i = 1
    while total_volume < volume_of_fluid_target and i < N:
        # Update forces for plunger/barrel
        B_DD[-2] = k*l - A_p*eta/L*(v[-1, i-1]-v[-2, i-1])
        B_DD[-1] = A_p*eta/L*(v[-1, i-1]-v[-2, i-1])

        forces_if_stuck = A_DD @ u[:, i-1] + B_DD
        F_plunger_if_stuck = forces_if_stuck[-2]

        forces = forces_if_stuck.copy()
        if abs(F_plunger_if_stuck) >= F_friction:
            forces[-1] = F_plunger_if_stuck - np.sign(F_plunger_if_stuck)*F_friction

        # Solve (diagonal mass -> division)
        a[:, i] = forces / np.diag(m)
        v[:, i] = v[:, i-1] + dt * a[:, i]
        u[:, i] = u[:, i-1] + dt * v[:, i]

        v[0, i] = 0
        u[0, i] = 0

        # Fluid injection
        Q = A_p * v[-2, i-1]
        delta_p = 8 * mu * Ln * Q / (np.pi * r**4)
        if Q > 0:
            total_volume += Q * dt

        i += 1

    return total_volume

# --- Run simulations ---
for ki, k_val in enumerate(k_values):
    for li, Ln_val in enumerate(Ln_values):
        fluid_injected[ki, li] = simulate_injection(k_val, Ln_val)

# --- Plotting ---
plt.figure(figsize=(8,6))
plt.imshow(fluid_injected, extent=[Ln_values[0]*1e3, Ln_values[-1]*1e3,
                                   k_values[0]/(n-2), k_values[-1]/(n-2)],
           origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Fluid injected (m³)')
plt.xlabel('Needle length Ln (mm)')
plt.ylabel('Spring constant per spring k (N/m)')
plt.title('Fluid injected vs spring constant and needle length')
plt.show()
