import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# for simple 3 mass system
n = 100
k =  0.3 # was 10
l = 60  # was 1
m = 1.8e-3/n # was 0.5
M = 0.02 # was 5 need to make this more accurate but its ok for now
ds = 2 # was 2
df = ds + 2.5 # was 3
dm = df + 10 # was 4 this is waff
di = 16 # was 4.5
mu_s = 2 # this was 1, need to reference that another time
mu_f = 0.3 # this was 0.5 need to reference
mu_m = 1.2 # was 2, need to reference this
spring_length = 35 # was 1 complete waff

# plunger part
Nf = 1
mu_k = 1
C_p = 1
A_p = 1
delta_p = 1
volume_of_fluid = 1


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


for i in range(1, N): # keep track of i for future for loops
    if u[-1, i-1] < ds:
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i-1]*dt + v[:, i - 1]
        u[:, i] = v[:,i-1]*dt + u[:, i-1]
        u[0, i] = 0
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
    else:
        break

end_phase3 = i

# plunger part

Q = A_p*v[-1, i - 1]

for i in range(end_phase3, N): # keep track of i for future for loops
    if Q*i*dt < volume_of_fluid: # change condition to something abt v ? ?
        B_DD[-1] = k*l + Nf*mu_k + A_p*delta_p + C_p*v[-1, i - 1]
        S = 1/m * ((A_DD@u)[:, i-1] + B_DD)
        S[-1] = S[-1]*m/M
        a[:, i] = S
        v[:, i] = a[:, i-1]*dt + v[:, i - 1]
        u[:, i] = v[:,i-1]*dt + u[:, i-1]
        u[0, i] = 0
        Q = A_p * v[-1, i - 1]
    else:
        break

end_phase4 = i



# plotting

u = u[:, 0:end_phase4]
v = v[:, 0:end_phase4]
a = a[:, 0:end_phase4]

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

plt.plot([i for i in range(end_phase4)], u[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Displacement of syringe")
plt.show()

print(len(u[-1, :]), len(v[-1, :]))

plt.plot([i for i in range(end_phase4)], v[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Velocity of syringe")
plt.show()

plt.plot([i for i in range(end_phase4)], a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Acceleration of syringe")
plt.show()

plt.plot([i for i in range(end_phase4)], M*a[-1, :])
plt.xlabel("Time in seconds")
plt.ylabel("Force of syringe")
plt.show()