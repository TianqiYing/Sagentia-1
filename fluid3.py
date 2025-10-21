import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================================================================
# A) Penetration (spring–chain) model + visualization
# ================================================================
def run_penetration_visual(
        n=50, k=10.0, l=1.0, m=1.0, M=5.0,
        ds=2.0, df=3.0, dm=4.0, di=4.5,
        mu_s=1.0, mu_f=1.0, mu_m=1.5,
        spring_length=1.0, dt=0.002, N=2500):
    """
    Simulate and visualize the penetration process of a spring–mass chain.
    Returns: time array, spring-end force (positive = push), and t0 (handover time)
    """

    # Matrix and boundary initialization
    A_DD = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            A_DD[i, i], A_DD[i, i+1] = -k, k
        elif i == n-1:
            A_DD[i, i-1], A_DD[i, i] = k, -k
        else:
            A_DD[i, i-1], A_DD[i, i], A_DD[i, i+1] = k, -2*k, k
    B_DD = np.zeros(n); B_DD[0], B_DD[-1] = -k*l, k*l

    # Initialization
    u = np.zeros((n, N)); v = np.zeros((n, N))
    u[:, 0] = np.linspace(0, spring_length, n)
    Fspring = []
    t_arr = np.arange(N) * dt

    # Time marching
    i = 1
    while i < N and u[-1, i-1] < di:
        S = 1/m * ((A_DD @ u)[:, i-1] + B_DD); S[-1] *= m / M
        v[:, i] = v[:, i-1] + S*dt
        u[:, i] = u[:, i-1] + v[:, i-1]*dt
        u[0, i] = 0.0
        # Note: negative displacement difference means compression, so flip sign to get positive push
        Fspring.append(-k * (u[-2, i] - u[-1, i]))
        i += 1

    t_arr = t_arr[:i]
    Fspring = np.asarray(Fspring[:i])

    # Visualization: deformation of spring + force evolution
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(u[:, ::int(i/20)], np.arange(n)[:,None], lw=1)
    plt.gca().invert_yaxis()
    plt.title("Spring–mass deformation over time")
    plt.xlabel("Displacement"); plt.ylabel("Mass index")

    plt.subplot(1,2,2)

    min_len = min(len(t_arr), len(Fspring))
    t_arr = t_arr[:min_len]
    Fspring = Fspring[:min_len]

    plt.plot(t_arr, Fspring, color='C0')
    plt.title("Spring force during penetration")
    plt.xlabel("Time (s)"); plt.ylabel("Force (N)")
    plt.tight_layout()
    plt.show()

    return t_arr, Fspring, t_arr[-1]


# ================================================================
# B) Injection (plunger–hydraulic ODE, fixed sign)
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

def F_seal(v):
    if abs(v) < 1e-12:
        return 0.0
    return np.sign(v)*(F_k + (F_s-F_k)*np.exp(-abs(v)/v0))

# ================================================================
# C) Coupled fast Euler integration
# ================================================================
def main():
    print("Running coupled injector simulation with corrected signs...")
    t_pen, F_pen, t0 = run_penetration_visual()
    min_len = min(len(t_pen), len(F_pen))
    t_pen, F_pen = t_pen[:min_len], F_pen[:min_len]

    # Spring force continuation
    tau_s = 8e-3
    Fs_end = F_pen[-1]
    interp = interp1d(t_pen, F_pen, kind="linear",
                      bounds_error=False, fill_value=(F_pen[0], F_pen[-1]))

    def Fspring(t):
        if t <= t0:
            return float(interp(t))
        else:
            return float(Fs_end * np.exp(-(t - t0)/tau_s))

    # Explicit Euler integration for injection phase
    dt = 2e-4
    T_total = t0 + 0.06
    steps = int((T_total - t0)/dt)
    t = np.linspace(t0, T_total, steps)
    u_p = np.zeros(steps); v_p = np.zeros(steps); p = np.zeros(steps)

    for i in range(1, steps):
        Fs = Fspring(t[i])
        du = v_p[i-1]
        # corrected sign convention: pressure resists motion
        dv = (Fs - (F_seal(v_p[i-1]) + c_p*v_p[i-1]) - A_p*p[i-1]) / M
        dp = (A_p*v_p[i-1] - p[i-1]/R_h) / C
        u_p[i] = u_p[i-1] + du*dt
        v_p[i] = v_p[i-1] + dv*dt
        p[i] = p[i-1] + dp*dt
        if i % 200 == 0:
            print(f"Progress: {i/steps*100:5.1f}%", end="\r")

    Q_out = p / R_h
    V = np.cumsum(Q_out * dt)
    done_idx = np.argmax(V >= V_target) if np.any(V >= V_target) else len(t)-1
    T_done = t[done_idx]
    Pmax_bar = np.max(p)/1e5

    # Regression for Poiseuille validation
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
    # Plots for injection phase
    # ================================================================
    plt.figure(figsize=(12,9))
    # (1) Dynamic forces
    ax1 = plt.subplot(2,2,1)
    ax1.plot(t_pen, F_pen, color="C0", label="Spring force (penetration)")
    ax1.plot(t, [Fspring(tt) for tt in t], color="C1", linestyle='--', label="Spring force (injection)")
    ax1.plot(t, A_p*p, color="C3", linestyle='-', label="Hydraulic load A_p p")
    ax1.axvline(t0, color="gray", linestyle=":")
    ax1.axvline(T_done, color="k", linestyle="-.")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Force (N)")
    ax1.legend(); ax1.set_title("Dynamic forces at plunger end")

    # (2) Pressure and flow
    ax2 = plt.subplot(2,2,2)
    ax2.plot(t, p/1e5, color="C1", label="Pressure (bar)")
    ax2b = ax2.twinx()
    ax2b.plot(t, Q_out*1e6, color="C2", label="Flow (µL/s)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Pressure (bar)")
    ax2b.set_ylabel("Flow (µL/s)")
    ax2.legend(loc="upper left")
    ax2b.legend(loc="lower right")
    ax2.set_title("Pressure and flow vs time")

    # (3) Volume
    ax3 = plt.subplot(2,2,3)
    ax3.plot(t, V*1e6, color="C5", label="Injected volume (µL)")
    ax3.axhline(V_target*1e6, color="k", linestyle="--")
    ax3.axvline(T_done, color="k", linestyle="-.")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Volume (µL)")
    ax3.legend(); ax3.set_title("Cumulative injected volume")

    # (4) Poiseuille check
    ax4 = plt.subplot(2,2,4)
    ax4.scatter(p/1e5, Q_out*1e6, s=8, alpha=0.6, label="Data", color="C4")
    p_line = np.linspace(0, np.max(p)/1e5, 60)
    q_line = (p_line*1e5)/R_h*1e6
    ax4.plot(p_line, q_line, color="C0", label="Q = p / R_h")
    ax4.set_xlabel("Pressure (bar)"); ax4.set_ylabel("Flow (µL/s)")
    ax4.legend(); ax4.set_title(f"Poiseuille verification (R²={R2:.3f})")

    plt.tight_layout(); plt.show()

# ================================================================
if __name__ == "__main__":
    main()
