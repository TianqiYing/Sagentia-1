import numpy as np
import matplotlib.pyplot as plt

# ========== 1) Physical parameters ==========
# Spring parameters (from real EpiPen data)
k_total = 300.0        # N/m (0.3 N/mm)
u0_release = 0.03      # m, initial compression (30 mm)
F_preload = 12.0       # N, preload
m_spring = 0.0018      # kg, total spring mass (1.8 g)

# Discrete spring chain
n = 10
m_seg = m_spring / n             # mass per segment
k_seg = k_total / n              # per-segment stiffness (ensures overall k_total)
damp_ratio = 0.01                # small damping for stability

# Piston and fluid
M = 0.005                        # kg, plunger + liquid mass
Dp = 8.65e-3                     # m, piston diameter
A_p = np.pi * (Dp / 2) ** 2      # m², piston area
di = 0.27e-3                     # m, needle inner diameter (0.27 mm)
rn = di / 2                      # m, radius
Ln = 18e-3                       # m, length of needle
mu = 8.8e-3                      # Pa·s, fluid viscosity (8.8 cSt)
rho = 1000.0                     # kg/m³
R_h = 8 * mu * Ln / (np.pi * rn ** 4)  # Poiseuille resistance
C = 2e-9                         # m³/Pa, compliance
F_s, F_k, v0 = 8.0, 5.0, 5e-3    # sealing friction parameters
c_p = 0.0                        # additional viscous damping

# Numerical settings
dt = 2e-4                        # s, time step (stable)
T_total = 0.06                   # s
N = int(T_total / dt)
inject_start = int(0.015 / dt)   # start injection after 15 ms

# ========== 2) Initialization ==========
u = np.zeros((n, N))      # displacement
v = np.zeros((n, N))      # velocity
a = np.zeros((n, N))      # acceleration

# Discrete Laplacian for the spring chain
A = np.zeros((n, n))
for i in range(1, n - 1):
    A[i, i - 1], A[i, i], A[i, i + 1] = 1, -2, 1
A[0, 0], A[0, 1] = -1, 1
A[-1, -2], A[-1, -1] = 1, -1

# External displacement boundary (left end fixed compression)
B = np.zeros(n)
B[0] = -u0_release

# Data records
Fspring = np.zeros(N)
Fload = np.zeros(N)
p_hist = np.zeros(N)
Qin = np.zeros(N)
Qout = np.zeros(N)
V_hist = np.zeros(N)
Re_hist = np.zeros(N)

# ========== 3) Simulation ==========
p = 0.0
V = 0.0
done_idx = None

for i in range(1, N):
    # Internal spring chain dynamics
    S = (k_seg / m_seg) * (A @ u[:, i - 1] + B) - damp_ratio * v[:, i - 1]

    if i < inject_start:
        # Spring release phase
        a[:, i] = S
        a[-1, i] = S[-1] * (m_seg / M) + F_preload / M
    else:
        # Injection phase
        v_end = v[-1, i - 1]
        sign_v = 0.0 if abs(v_end) < 1e-9 else np.sign(v_end)
        F_seal = (F_k + (F_s - F_k) * np.exp(-abs(v_end) / v0)) * sign_v

        # Pressure evolution: C dp/dt = A*v - p/Rh
        dp = (A_p * v_end - p / R_h) / C * dt
        p = max(0.0, p + dp)

        # Forces
        F_spr_end = k_seg * (u[-2, i - 1] - u[-1, i - 1])
        F_hydro = A_p * p
        F_ld = F_hydro + F_seal + c_p * v_end

        a[:, i] = S
        a[-1, i] = (F_spr_end - F_ld) / M

        # Records
        Fspring[i] = F_spr_end
        Fload[i] = F_ld
        p_hist[i] = p
        Qin[i] = A_p * v_end
        Qout[i] = p / R_h
        V += Qin[i] * dt
        V_hist[i] = V

        v_mean = Qout[i] / (np.pi * rn ** 2 + 1e-30)
        Re_hist[i] = rho * v_mean * (2 * rn) / mu

        if done_idx is None and V >= 0.3e-3:
            done_idx = i

    # Integration (semi-implicit)
    v[:, i] = v[:, i - 1] + a[:, i - 1] * dt
    u[:, i] = u[:, i - 1] + v[:, i] * dt
    u[0, i] = 0.0

if done_idx is None:
    done_idx = N - 1

t = np.arange(N) * dt

# ========== 4) Linear fit for Poiseuille validation ==========
mask = (np.arange(N) >= inject_start)
pp = p_hist[mask]
qq = Qout[mask]
Afit = np.vstack([pp, np.ones_like(pp)]).T
s_fit, b_fit = np.linalg.lstsq(Afit, qq, rcond=None)[0]
yhat = s_fit * pp + b_fit
SS_res = np.sum((qq - yhat) ** 2)
SS_tot = np.sum((qq - np.mean(qq)) ** 2) + 1e-30
R2 = 1.0 - SS_res / SS_tot
s_theory = 1.0 / R_h
Re_max = np.max(Re_hist)
Pmax_bar = np.max(p_hist) / 1e5
T_done_ms = t[done_idx] * 1e3

print(f"Injection completed at {T_done_ms:.1f} ms, "
      f"Pmax={Pmax_bar:.3f} bar, Re_max={Re_max:.0f}, R²={R2:.4f}")

# ========== 5) Plotting ==========
plt.figure(figsize=(12, 9))

# (1) Forces
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t, Fspring, label="Spring force", color="C0")
ax1.plot(t, Fload, "--", label="Load force", color="C3")
ax1.axvline(inject_start * dt, color="gray", linestyle=":")
ax1.axvline(t[done_idx], color="k", linestyle="-.")
ax1.set_title("Dynamic forces at plunger end")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force (N)")
ax1.legend()

# (2) Pressure & Flow
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t, p_hist / 1e5, color="C1", label="Pressure (bar)")
ax2.axvline(inject_start * dt, color="gray", linestyle=":")
ax2.axvline(t[done_idx], color="k", linestyle="-.")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Pressure (bar)", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
ax2b = ax2.twinx()
ax2b.plot(t, Qin * 1e6, color="C2", label="Flow in (μL/s)")
ax2b.plot(t, Qout * 1e6, "--", color="C4", label="Flow out (μL/s)")
ax2b.set_ylabel("Flow rate (μL/s)")
ax2b.legend(loc="lower right")
ax2.set_title("Pressure and flow vs. time")

# (3) Volume
ax3 = plt.subplot(2, 2, 3)
ax3.plot(t, V_hist * 1e6, color="C5", label="Injected volume (μL)")
ax3.axhline(300, color="k", linestyle="--", linewidth=1)
ax3.axvline(t[done_idx], color="k", linestyle="-.")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Volume (μL)")
ax3.legend()
ax3.set_title(f"Cumulative volume (complete at {T_done_ms:.1f} ms)")

# (4) Q–p linearity
ax4 = plt.subplot(2, 2, 4)
ax4.scatter(p_hist[mask] / 1e5, Qout[mask] * 1e6, s=8, color="C4", alpha=0.6, label="Sim data")
p_line = np.linspace(0, max(p_hist[mask]) / 1e5 + 0.01, 50)
q_the = (p_line * 1e5) * s_theory * 1e6
ax4.plot(p_line, q_the, color="C0", label="Q = p / R_h (theory)")
q_fit = (p_line * 1e5) * s_fit * 1e6 + b_fit * 1e6
ax4.plot(p_line, q_fit, "--", color="C3", label=f"fit: R²={R2:.3f}")
ax4.set_xlabel("Pressure (bar)")
ax4.set_ylabel("Flow out (μL/s)")
ax4.set_title(f"Poiseuille check (Re_max={Re_max:.0f} < 2300)")
ax4.legend()

plt.tight_layout()
plt.show()
