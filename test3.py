import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# for simple 3 mass system
n = 100
k =  10
l = 1
m = 1
M = 5
ds = 2
df = 3
dm = 4
di = 4.5
mu_s = 1
mu_f = 1
mu_m = 1.5
spring_length = 1
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

dt = 0.01
N = 10000
initial_positions = np.linspace(0, spring_length, n)

u = np.zeros((n, N))
v = np.zeros((n, N))
a = np.zeros((n, N))

u[:, 0] = initial_positions

# things to track

resistive_forces = []

for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1] < ds:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i-1]*dt + v[:, i - 1]
        u[:, i] = v[:,i-1]*dt + u[:, i-1]
        u[0, i] = 0
        resistive_forces.append(0)
    else:
        break

end_phase1 = i


# part where it enters skin

f_max = 2
f_min = 1
u_puncture = 2.2
s = 0.5

for i in range(end_phase1, N):
    if u[-1, i-1] < df:
        z = s*(u[-1, i-1] - ds - u_puncture)
        A_DD[-1, -1] = -k - mu_s*np.tanh(z)
        B_DD[-1] = k * l + ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z))

        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i-1]*dt + v[:, i - 1]
        u[:, i] = v[:,i-1]*dt + u[:, i-1]
        u[0, i] = 0
        resistive_forces.append(ds*mu_s*(np.tanh(z)) + f_max*(1-np.tanh(z)) - mu_s*np.tanh(z))
    else:
        break

end_phase2 = i


# moving onto fat through to muscle


A_DD[-1, -1] = -k - mu_f
B_DD[-1] = k*l - mu_s*(df - ds) + mu_f*df

for i in range(end_phase2, N):
    if u[-1, i-1] < dm:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
        resistive_forces.append(- mu_s*(df - ds) + mu_f*df - mu_f)
    else:
        break

end_phase2 = i


# moving onto muscle through to point of injection

A_DD[-1, -1] = -k - mu_m
B_DD[-1] = k*l - mu_s*(df - ds) - mu_f*(dm - df) + mu_m*dm

for i in range(end_phase2, N):
    if u[-1, i-1] < di:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i - 1] * dt + v[:, i - 1]
        u[:, i] = v[:, i - 1] * dt + u[:, i - 1]
        u[0, i] = 0
        resistive_forces.append(- mu_s*(df - ds) - mu_f*(dm - df) + mu_m*dm - mu_m)
    else:
        break

end_phase3 = i
u = u[:, 0:end_phase3]
v = v[:, 0:end_phase3]
a = a[:, 0:end_phase3]

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

plt.plot([i for i in range(end_phase3)], u[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of syringe")
plt.show()

print(len(u[-1, :]), len(v[-1, :]))

plt.plot([i for i in range(end_phase3)], v[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of syringe")
plt.show()

plt.plot([i for i in range(end_phase3)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase3)], M*a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()


plt.plot([i for i in range(end_phase3-1)], resistive_forces)
plt.show()
