import numpy as np
import matplotlib.pyplot as plt


n = 10 # number of spring mass pairs
dt = 0.1

A_DD = np.zeros((n, n))
spring_length = 5 # cm
natural_length = 11.73
m = 1# mass for spring mass chain in kg
syringe_mass = 1# kg
large_mass = 1000
dl = 0.5

for _ in range(0, n):
        # last row
    if _ == n - 1:
        A_DD[_, _] = 1
        A_DD[_, _-1] = -1

    elif _ == 0:
        A_DD[_, _] = 1
        A_DD[_, _+1] = -1

        # other rows
    else:
        A_DD[_, _ - 1] = -1
        A_DD[_, _] = 2
        A_DD[_, _ + 1] = 1
k = 1
A_DD = k*A_DD
initial_positions = np.linspace(0, spring_length, n)
u = np.full(n, fill_value=0.0)
B_DD = np.zeros(n)
B_DD[-1] = -natural_length*k
B_DD[0] = natural_length*k
M = np.diag(np.full(n, m))
M[-1, -1] = syringe_mass
M[0, 0] = large_mass
d1u = np.zeros(n)

while u[-1] < dl:
    d2u = np.linalg.solve(M, -A_DD@u - B_DD)
    d1u += d2u * dt
    u += d1u * dt

print(u)
final_positions = initial_positions + u
plt.plot(final_positions, np.zeros_like(final_positions), 'o-', label="Masses")
plt.xlabel("Position along spring")
plt.title("Final Positions of Spring Mass Chain")
plt.grid(True)
plt.legend()
plt.show()

# moving on to second part of ode once it is in the skin

mu_s = 1 # friction coefficient for skin
d0 = 10

for _ in range(0, n):
        # last row
    if _ == n - 1:
        A_DD[_, _] = 1
        A_DD[_, _-1] = -1

    elif _ == 0:
        A_DD[_, _] = -1
        A_DD[_, _+1] = 1

        # other rows
    else:
        A_DD[_, _ - 1] = 1
        A_DD[_, _] = -2
        A_DD[_, _ + 1] = 1

A_DD = k*A_DD
A_DD[-1, -1] += mu_s*d0
B_DD = np.zeros(n)
B_DD[-1] = -k*(natural_length) + mu_s*d0
M = np.diag(np.full(n, m))
M[-1, -1] = syringe_mass
M[0, 0] = large_mass

while u[-1] < dl:
    d2u = np.linalg.solve(M, -A_DD@u - B_DD)
    d1u += d2u * dt
    u += d1u * dt

final_positions +=  u
print(u)
plt.plot(final_positions, np.zeros_like(final_positions), 'o-', label="Masses")
plt.xlabel("Position along spring")
plt.title("Final Positions of Spring Mass Chain")
plt.grid(True)
plt.legend()
plt.show()


####################################################################################################
# moving on to third part of ode once its in the fat


mu_f = 0.5 # friction coefficient for skin
d1 = 10

for _ in range(0, n):
    if _ == 0:
        A_DD[_, _] = -2
        A_DD[_, _ + 1] = 1

        # last row
    elif _ == n - 1:
        A_DD[_, _] = -1 - mu_f*u[-1]/k

        # other rows
    else:
        A_DD[_, _ - 1] = 1
        A_DD[_, _] = -2
        A_DD[_, _ + 1] = 1
k = 15
A_DD = k*A_DD
B_DD = np.zeros(n)
B_DD[-1] = natural_length + mu_s*(d1-d0) + mu_f*(d1)
M = np.diag(np.full(n, m))
M[-1, -1] = syringe_mass

while u[-1] < dl:
    d2u = np.linalg.solve(M, -A_DD@u + B_DD)
    d1u += d2u * dt
    u += d1u * dt

final_positions += u

plt.plot(final_positions, np.zeros_like(final_positions), 'o-', label="Masses")
plt.xlabel("Position along spring")
plt.title("Final Positions of Spring Mass Chain")
plt.grid(True)
plt.legend()
plt.show()