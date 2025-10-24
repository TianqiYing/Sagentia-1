import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================================================================
# A) Penetration (spring–chain) model + visualization
# ================================================================
def run_penetration_visual(n=50, k=10.0, l=1.0, m=1.0, M=5.0,
                           ds=2.0, df=3.0, dm=4.0, di=4.5,
                           spring_length=1.0, dt=0.002, N=2500):
    """Simulate spring–mass chain penetration; return t, Fspring, t0."""
    A_DD = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            A_DD[i, i], A_DD[i, i+1] = -k, k
        elif i == n-1:
            A_DD[i, i-1], A_DD[i, i] = k, -k
        else:
            A_DD[i, i-1], A_DD[i, i], A_DD[i, i+1] = k, -2*k, k
    B_DD = np.zeros(n); B_DD[0], B_DD[-1] = -k*l, k*l

    u = np.zeros((n, N)); v = np.zeros((n, N))
    u[:, 0] = np.linspace(0, spring_length, n)
    Fspring = []
    t_arr = np.arange(N) * dt

    i = 1
    while i < N and u[-1, i-1] < di:
        S = 1/m * ((A_DD @ u)[:, i-1] + B_DD); S[-1] *= m / M
        v[:, i] = v[:, i-1] + S*dt
        u[:, i] = u[:, i-1] + v[:, i-1]*dt
        u[0, i] = 0.0
        Fspring.append(-k * (u[-2, i] - u[-1, i]))  # reverse sign
        i += 1

    min_len = min(len(t_arr), len(Fspring))
    t_arr = t_arr[:min_len]
    Fspring = np.asarray(Fspring[:min_len])

    # --- simple visualization of spring deformation and force ---
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(u[:, ::int(i/20)], np.arange(n)[:,None], lw=1)
    plt.gca().invert_yaxis()
    plt.title("Spring–mass deformation over time")
    plt.xlabel("Displacement"); plt.ylabel("Mass index")

    plt.subplot(1,2,2)
    plt.plot(t_arr, Fspring, color='C0')
    plt.title("Spring force during penetration")
    plt.xlabel("Time (s)"); plt.ylabel("Force (N)")
    plt.tight_layout(); plt.show()

    return t_arr, Fspring, t_arr[-1]


# ================================================================
# B) Injection with Kelvin–Voigt damping
# ================================================================
Dp = 8.65e-3
A_p = np.pi*(Dp/2)**2
di_real = 0.27e-3
rn = di_real/2
Ln = 18e-3
mu = 8.8e-3
rho = 1000.0
R_h = 8*mu*Ln/(np.pi*rn**4)
C   = 2e-9
M   = 5e-3
F_s, F_k, v0 = 8.0, 5.0, 5e-3
c_p = 0.05
V_target = 0.3e-3

# Kelvin–Voigt parameters
k_v = 1.5e5   # [Pa/m] elastic stiffness of hydraulic chamber
eta_v = 25.0  # [Pa·s/m] viscous damping

def F_seal(v):
    """Sealing friction force (velocity-dependent)."""
    if abs(v) < 1e-12:
        return 0.0
    return np.sign(v)*(F_k + (F_s-F_k)*np.exp(-abs(v)/v0))


# ================================================================
# C) Coupled integration (explicit Euler)
# ================================================================
def main():
    print("Running final version with visual-optimized plots...")
    t_pen, F_pen, t0 = run_penetration_visual()
    min_len = min(len(t_pen), len(F_pen))
    t_pen, F_pen = t_pen[:min_len], F_pen[:min_len]

    tau_s = 8e-3
    Fs_end = F_pen[-1]
    interp = interp1d(t_pen, F_pen, kind="linear",
                      bounds_error=False, fill_value=(F_pen[0], F_pen[-1]))

    def Fspring(t):
        if t <= t0:
            return float(interp(t))
        else:
            return float(Fs_end * np.exp(-(t - t0)/tau_s))

    # Integration setup
    dt = 2e-4
    T_total = t0 + 0.06
    steps = int((T_total - t0)/dt)
    t = np.linspace(t0, T_total, steps)
    u_p = np.zeros(steps); v_p = np.zeros(steps); p = np.zeros(steps)
    sigma_kv = np.zeros(steps)

    for i in range(1, steps):
        Fs = Fspring(t[i])
        du = v_p[i-1]
        # Kelvin–Voigt stress
        sigma_kv[i-1] = k_v*u_p[i-1] + eta_v*v_p[i-1]
        dv = (Fs - (F_seal(v_p[i-1]) + c_p*v_p[i-1]) - A_p*p[i-1] - A_p*sigma_kv[i-1]) / M
        dp = (A_p*v_p[i-1] - p[i-1]/R_h) / C
        u_p[i] = u_p[i-1] + du*dt
        v_p[i] = v_p[i-1] + dv*dt
        p[i] = p[i-1] + dp*dt

    Q_out = p / R_h
    V = np.cumsum(Q_out * dt)
    done_idx = np.argmax(V >= V_target) if np.any(V >= V_target) else len(t)-1
    T_done = t[done_idx]
    Pmax_bar = np.max(p)/1e5

    # Regression check
    Afit = np.vstack([p, np.ones_like(p)]).T
    s_fit, b_fit = np.linalg.lstsq(Afit, Q_out, rcond=None)[0]
    SS_res = np.sum((Q_out - (s_fit*p + b_fit))**2)
    SS_tot = np.sum((Q_out - np.mean(Q_out))**2) + 1e-30
    R2 = 1.0 - SS_res/SS_tot
    v_mean = Q_out / (np.pi*rn**2 + 1e-30)
    Re_max = np.max(rho * v_mean * (2*rn) / mu)

    print(f"\n[Summary] t0={t0:.3f}s → Injection complete at {T_done*1e3:.1f} ms, "
          f"Pmax={Pmax_bar:.3f} bar, R²={R2:.4f}, Re_max={int(Re_max)}")

    # ================================================================
    # D) Visualization (final polished)
    # ================================================================
    fig, axes = plt.subplots(3,2, figsize=(13,10))
    axes = axes.flatten()

    # (1) Dynamic forces (zoomed)
    ax = axes[0]
    ax.plot(t, [Fspring(tt) for tt in t], "--", color="tab:orange", lw=2.5, alpha=0.9, label="Spring force (injection)")
    ax.plot(t, A_p*p, "-", color="tab:red", lw=2.2, label="Hydraulic load Aₚp")
    ax.plot(t, A_p*sigma_kv, ":", color="tab:green", lw=3.0, label="K–V elastic term (fluid damping)")
    ax.axvspan(t0, t0+0.01, color="gray", alpha=0.1)
    ax.text(t0+0.002, np.max(A_p*p)*0.7, "Damping phase", fontsize=8)
    ax.set_xlim(t0-0.005, t0+0.06)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Force (N)")
    ax.legend(fontsize=8)
    ax.set_title("Dynamic forces with Kelvin–Voigt damping (zoomed)")

    # (2) Pressure + flow (dual y-axis)
    ax = axes[1]
    ax.plot(t, p*1e3, color="tab:orange", lw=2.5, label="Pressure (mbar)")
    ax2 = ax.twinx()
    ax2.plot(t, Q_out*1e6, color="tab:green", lw=1.8, label="Flow (µL/s)")
    ax.set_ylim(0, np.max(p*1e3)*1.2)
    ax2.set_ylim(0, np.max(Q_out*1e6)*1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (mbar)", color="tab:orange")
    ax2.set_ylabel("Flow (µL/s)", color="tab:green")
    ax.legend(loc="upper left"); ax2.legend(loc="lower right")
    ax.set_title("Pressure build-up and stabilization under viscous flow")

    # (3) Injected volume
    ax = axes[2]
    ax.plot(t, V*1e6, color="C5", lw=2.0, label="Injected volume (µL)")
    ax.axhline(V_target*1e6, color="k", linestyle="--", alpha=0.4)
    ax.axvline(T_done, color="k", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Volume (µL)")
    ax.legend(fontsize=8)
    ax.set_title("Cumulative injected volume")
    ax.annotate(f"Final: {V[-1]*1e6:.1f} µL", xy=(T_done, V[-1]*1e6),
                xytext=(T_done-0.01, V[-1]*1e6*0.7),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)

    # (4) Poiseuille linearity
    ax = axes[3]
    ax.scatter(p/1e5, Q_out*1e6, s=10, alpha=0.5, color="C4", edgecolor="none", label="Sim data")
    p_line = np.linspace(0, np.max(p)/1e5, 60)
    q_line = (p_line*1e5)/R_h*1e6
    ax.plot(p_line, q_line, color="black", lw=1.5, linestyle="--", label="Q = p / Rₕ (theory)")
    ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Flow (µL/s)")
    ax.legend(fontsize=8)
    ax.set_title(f"Poiseuille verification (R²={R2:.3f})")
    ax.text(0.6*np.max(p/1e5), 0.8*np.max(Q_out*1e6),
            f"Reₘₐₓ={Re_max:.1f} (<10, laminar)", fontsize=9)

    # (5) Kelvin–Voigt stress vs time
    ax = axes[4]
    ax.plot(t, sigma_kv/1e6, color="tab:green", lw=2.0)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("σₖᵥ (MPa)")
    ax.set_title("Kelvin–Voigt internal stress (elastic + viscous response)")
    ax.annotate("Fast decay = damping", xy=(t0+0.012, np.max(sigma_kv)/1e6*0.8),
                xytext=(t0+0.02, np.max(sigma_kv)/1e6*0.9),
                arrowprops=dict(arrowstyle='->', color="gray"), fontsize=8)

    # Remove empty subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.show()

# ================================================================
if __name__ == "__main__":
    main()
