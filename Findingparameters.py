import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_injection(k_value, Ln_value):
    n = 10
    k = k_value*(n-2)
    l = 60e-3 / (n-1)
    mass = 1.8e-3 / (n-1)
    ds = 1e-3
    df = ds + 1.55e-3
    dm = df + 10e-3
    di = 16e-3
    mu_s = 5000
    mu_f = 20000
    mu_m = 100000
    spring_length = 0.035 / (n-2)

    # plunger part
    Mp = 0.0012
    Ms = 0.0022

    # Create mass matrix
    m = np.diag([mass]*(n-2) + [Mp] + [Ms])

    eta = 1e-3
    Nf = 2
    mu_k = 8
    C_p = 0.033
    A_p = 5.88e-5
    volume_of_fluid = 0.3e-3
    mu = 0.00089
    r = 0.135e-3
    L = 34e-3  # Distance between barrel and plunger for 2mL fluid

    f_max = 150000
    u_puncture = 2.2e-3
    s = 0.5

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
            A_DD[_, _] = 0
        else:
            A_DD[_, _] = -2*k
            A_DD[_, _-1] = k
            A_DD[_, _+1] = k

    B_DD = np.zeros(n)
    B_DD[0] = -k*l

    dt = 0.000001
    N = 10000

    u = np.zeros((n, N))
    v = np.zeros((n, N))
    a = np.zeros((n, N))

    # Phase 1: Before skin contact
    for i in range(1, N):
        if u[-1, i-1] < ds:
            B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
            B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

            forces = A_DD @ u[:, i - 1] + B_DD
            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]

            F_plunger = B_DD[-2]
            if abs(F_plunger) < F_friction:
                a[-1, i] = a[-2, i]
            else:
                a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

            v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
            u[-1, i] = u[-1, i - 1] + dt * v[-1, i]
            v[0, i] = 0
            u[0, i] = 0
        else:
            break
    end_phase1 = i

    # Phase 2: Skin penetration
    for i in range(end_phase1, N):
        if u[-1, i-1] < df:
            z = s*(u[-1, i-1] - ds - u_puncture)
            A_DD[-1, -1] = -k - mu_s*np.tanh(z)
            B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z))
            B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

            forces = A_DD @ u[:, i - 1] + B_DD
            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]

            F_plunger = B_DD[-2]
            if abs(F_plunger) < F_friction:
                a[-1, i] = a[-2, i]
            else:
                a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

            v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
            u[-1, i] = u[-1, i - 1] + dt * v[-1, i]
            v[0, i] = 0
            u[0, i] = 0
        else:
            break
    end_phase2 = i

    # Phase 3: Fat penetration
    for i in range(end_phase2, N):
        if u[-1, i-1] < dm:
            A_DD[-1, -1] = -k - mu_f
            B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) + mu_f * df
            B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

            forces = A_DD @ u[:, i - 1] + B_DD
            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]

            F_plunger = B_DD[-2]
            if abs(F_plunger) < F_friction:
                a[-1, i] = a[-2, i]
            else:
                a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

            v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
            u[-1, i] = u[-1, i - 1] + dt * v[-1, i]
            v[0, i] = 0
            u[0, i] = 0
        else:
            break
    end_phase3 = i

    # Phase 4: Muscle penetration
    for i in range(end_phase3, N):
        if u[-1, i-1] < di:
            B_DD[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) - mu_s * (df - ds) - mu_f * (dm - df) + mu_m * dm
            B_DD[-2] = k*l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])
            A_DD[-1, -1] = -k - mu_m

            forces = A_DD @ u[:, i - 1] + B_DD
            a[:, i] = np.linalg.solve(m, forces)
            v[:, i] = v[:, i - 1] + dt * a[:, i]
            u[:, i] = u[:, i - 1] + dt * v[:, i]

            F_plunger = B_DD[-2]
            if abs(F_plunger) < F_friction:
                a[-1, i] = a[-2, i]
            else:
                a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

            v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
            u[-1, i] = u[-1, i - 1] + dt * v[-1, i]
            v[0, i] = 0
            u[0, i] = 0
        else:
            break
    end_phase4 = i

    # Phase 5: Fluid injection
    Q = A_p * v[-1, i - 1]
    total_volume = 0

    for i in range(end_phase4, N):
        if total_volume < volume_of_fluid:
            if Q > 0:
                total_volume += Q * dt

            delta_p = 8 * mu * Ln_value * Q / (np.pi * r**4)

            B_DD_internal = B_DD.copy()
            B_DD_internal[-1] = A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1]) + Nf * mu_k + A_p * delta_p + C_p * v[-1, i - 1]
            B_DD_internal[-2] = k * l - A_p * eta / L * (v[-1, i - 1] - v[-2, i - 1])

            a_internal = np.linalg.solve(m[:-1, :-1], A_DD[:-1, :-1] @ u[:-1, i - 1] + B_DD_internal[:-1])
            v[:-1, i] = v[:-1, i - 1] + dt * a_internal
            u[:-1, i] = u[:-1, i - 1] + dt * v[:-1, i]
            a[:-1, i] = a_internal

            F_plunger = B_DD_internal[-2]
            if abs(F_plunger) < F_friction:
                a[-1, i] = a[-2, i]
            else:
                a[-1, i] = (F_plunger - np.sign(F_plunger) * F_friction) / Ms

            v[-1, i] = v[-1, i - 1] + dt * a[-1, i]
            u[-1, i] = u[-1, i - 1] + dt * v[-1, i]
            v[0, i] = 0
            u[0, i] = 0

            Q = A_p * v[-1, i - 1]
        else:
            break

    return total_volume

# Parameter ranges
k_values = np.linspace(1500, 15000, 8)  # Spring constant range (N/m)
Ln_values = np.linspace(0.010, 0.025, 8)  # Needle length range (m)

# Create meshgrid
K, Ln = np.meshgrid(k_values, Ln_values)
fluid_injected = np.zeros_like(K)

print("Running parameter sweep...")
total_simulations = len(k_values) * len(Ln_values)
current_sim = 0

for i in range(len(k_values)):
    for j in range(len(Ln_values)):
        current_sim += 1
        fluid_injected[j, i] = simulate_injection(K[j, i], Ln[j, i])
        print(f"Progress: {current_sim}/{total_simulations} - "
              f"k={K[j, i]:.0f} N/m, Ln={Ln[j, i]*1000:.1f} mm -> "
              f"{fluid_injected[j, i]*1e6:.2f} µL injected")

# Create 3D plot
fig = plt.figure(figsize=(14, 10))

# 3D Surface plot
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(K/1000, Ln*1000, fluid_injected*1e6,
                       cmap='viridis', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Spring Constant (kN/m)')
ax1.set_ylabel('Needle Length (mm)')
ax1.set_zlabel('Fluid Injected (µL)')
ax1.set_title('3D Surface: Fluid Injection vs Parameters')
fig.colorbar(surf, ax=ax1, shrink=0.6, label='Fluid Injected (µL)')

# Contour plot
ax2 = fig.add_subplot(222)
contour = ax2.contourf(K/1000, Ln*1000, fluid_injected*1e6, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='Fluid Injected (µL)')
target_volume = 300  # 0.3 mL = 300 µL
CS = ax2.contour(K/1000, Ln*1000, fluid_injected*1e6,
                levels=[target_volume], colors='red', linewidths=3)
ax2.clabel(CS, inline=True, fontsize=10, fmt='%d µL')
ax2.set_xlabel('Spring Constant (kN/m)')
ax2.set_ylabel('Needle Length (mm)')
ax2.set_title('Contour Plot\n(Red line = target 300 µL)')
ax2.grid(True, alpha=0.3)

# Spring constant slice
ax3 = fig.add_subplot(223)
for j in range(len(Ln_values)):
    ax3.plot(K[0,:]/1000, fluid_injected[j,:]*1e6, 'o-',
             label=f'Ln={Ln_values[j]*1000:.1f}mm')
ax3.axhline(y=300, color='red', linestyle='--', linewidth=2, label='Target (300 µL)')
ax3.set_xlabel('Spring Constant (kN/m)')
ax3.set_ylabel('Fluid Injected (µL)')
ax3.set_title('Effect of Spring Constant')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.grid(True, alpha=0.3)

# Needle length slice
ax4 = fig.add_subplot(224)
for i in range(len(k_values)):
    ax4.plot(Ln[:,0]*1000, fluid_injected[:,i]*1e6, 's-',
             label=f'k={k_values[i]/1000:.1f}kN/m')
ax4.axhline(y=300, color='red', linestyle='--', linewidth=2, label='Target (300 µL)')
ax4.set_xlabel('Needle Length (mm)')
ax4.set_ylabel('Fluid Injected (µL)')
ax4.set_title('Effect of Needle Length')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print optimal parameters
max_volume = np.max(fluid_injected)
max_idx = np.unravel_index(np.argmax(fluid_injected), fluid_injected.shape)
optimal_k = K[max_idx] / 1000  # kN/m
optimal_Ln = Ln[max_idx] * 1000  # mm

print(f"\n=== RESULTS SUMMARY ===")
print(f"Maximum fluid injected: {max_volume*1e6:.2f} µL")
print(f"Optimal spring constant: {optimal_k:.1f} kN/m")
print(f"Optimal needle length: {optimal_Ln:.1f} mm")
print(f"Target volume: 300 µL")
print(f"Achieved: {max_volume*1e6:.2f} µL ({max_volume/0.3e-3*100:.1f}% of target)")