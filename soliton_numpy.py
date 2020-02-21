import time
import numpy as np
from functools import reduce

tstart = time.time()
pi = np.pi


def chebp(n):
    """
    chebyshev points
    """
    return np.cos(pi * np.arange(n) / (n - 1.))


def cheb(n, l):
    """
    chebyshev derivative matrix
    input dimension N and length of interval
    """
    y = chebp(n)
    c = np.ones(n)
    c[0] = c[-1] = 2.
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = (2. / l) * ((c[i] / c[j]) * (-1) ** (i + j)) / (y[i] - y[j])
    return d - np.diag(np.sum(d, axis=1))


# Parameters
mu = 5.
eps = 1e-6
do = 1
k, kmax = 0, 10
Lz, Lx = 1., 24.
Nz, Nx = 30, 60
N = Nz * Nx
dz = cheb(Nz, Lz)
z = (chebp(Nz) + 1.) / 2.
dx = cheb(Nx, Lx)
x = (Lx / 2.) * chebp(Nx)
Ez, Ex, E = np.eye(Nz), np.eye(Nx), np.eye(N)
Dz = np.kron(dz, Ex)
Dx = np.kron(Ez, dx)
Dzz = np.kron(dz @ dz, Ex)
Dxx = np.kron(Ez, dx @ dx)
xx, zz = np.meshgrid(x, z)  # be careful of the order!!!
Z = zz.flatten()
X = xx.flatten()
z0 = np.nonzero(Z == 0.)
z1 = np.nonzero(Z == 1.)
xb = np.nonzero(np.abs(X) == Lx / 2.)
border = reduce(np.union1d, (z0[0], z1[0], xb[0]))
psi = 4. * Z * np.tanh(X)
phi = mu * (1. - Z)
seed = np.concatenate([psi, phi])
fz = 1. - Z ** 3
fzp = -3. * (Z ** 2)
# %% Loop
while do == 1 and k < kmax:
    b1 = -phi ** 2 * psi + fz * (Z * psi - (Dxx @ psi) - fzp * (Dz @ psi) - fz * (Dzz @ psi))
    b2 = 2. * phi * psi ** 2 - (Dxx @ phi) - fz * (Dzz @ phi)
    b1[z1] = fzp[z1] * (Dz[z1] @ psi) + (Dxx[z1] @ psi) - psi[z1]
    b1[z0] = psi[z0]
    b1[xb] = Dx[xb] @ psi
    b2[z1] = phi[z1]
    b2[z0] = phi[z0] - mu
    b2[xb] = Dx[xb] @ phi
    b = - np.concatenate([b1, b2])
    A11 = np.diag(Z * fz - phi ** 2) - (fz * Dxx.T).T - (fz ** 2 * Dzz.T).T - (fz * fzp * Dz.T).T
    A12 = np.diag(-2. * phi * psi)
    A21 = np.diag(4. * psi * phi)
    A22 = np.diag(2. * psi ** 2) - Dxx - (fz * Dzz.T).T
    A11[z0] = E[z0]
    A11[z1] = (fzp[z1] * Dz[z1].T).T + Dxx[z1] - E[z1]
    A11[xb] = Dx[xb]
    A12[border] = A21[border] = 0.
    A22[z0] = E[z0]
    A22[z1] = E[z1]
    A22[xb] = Dx[xb]
    A = np.block([[A11, A12], [A21, A22]])
    sol = np.linalg.solve(A, b)
    normsol = np.linalg.norm(sol)

    if normsol < eps:
        do = 0

    seed = seed + sol
    psi, phi = np.split(seed, 2)
    k = k + 1
# %% Visualization

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(zz, xx, psi.reshape((Nz, Nx)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
tstop = time.time()
t = tstop - tstart
print(t)
