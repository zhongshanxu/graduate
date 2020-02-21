import time
import numpy as np
import torch as pt
import math

# choose cpu or gpu computing
dvc = pt.device("cuda")
# tune default type
pt.set_default_dtype(pt.double)

pt.cuda.synchronize()
t0 = time.time()
pi = math.pi


def kron(m1, m2):
    """
    matrix kronecker product
    """
    return pt.einsum("ij,kl->ikjl", m1, m2).view(m1.size(0) * m2.size(0), m1.size(1) * m2.size(1))


def chebp(n):
    """
    :param n: dimension
    :return: chebyshev points
    """
    return pt.cos(pi * pt.arange(n, dtype=pt.float64, device=dvc) / (n - 1.))


def cheb(n, l):
    """
    :param n: dimension
    :param l: length of internal
    :return: chebyshev derivative matrix
    """
    y = chebp(n)
    c = pt.ones(n, device=dvc)
    c[0] = c[-1] = 2.
    d = pt.zeros(n, n, device=dvc)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = (2. / l) * ((c[i] / c[j]) * (-1) ** (i + j)) / (y[i] - y[j])
    return d - pt.diag(pt.sum(d, 1))


def fourier(n, l):
    """
    :param n: dimension
    :param l: lenth of internal
    :return: fourier derivative matrix
    """
    d = pt.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                if n % 2 == 0:
                    d[i, j] = (pi / l) * (-1) ** (i - j) / pt.tan(pi * (i - j) / n)
                else:
                    d[i, j] = (pi / l) * (-1) ** (i - j) / pt.sin(pi * (i - j) / n)
    return d
# %% Parameters and initialization
mu = 5.
eps = 1e-6
do = 1
Lz, Lx = 1., 24.
Nz, Nx = 40, 80
N = Nz * Nx
k, kmax = 0, 10
dz = cheb(Nz, Lz)
z = (chebp(Nz) + 1.) / 2.
dx = cheb(Nx, Lx)
x = (Lx / 2.) * chebp(Nx)
Ez, Ex, E = pt.eye(Nz, device=dvc), pt.eye(Nx, device=dvc), pt.eye(N, device=dvc)
Dz = kron(dz, Ex)
Dx = kron(Ez, dx)
Dzz = kron(dz @ dz, Ex)
Dxx = kron(Ez, dx @ dx)
zz, xx = pt.meshgrid(z, x)  # be careful of the order!!!
Z = zz.flatten()
X = xx.flatten()
z0 = pt.nonzero(Z == 0.).squeeze()
z1 = pt.nonzero(Z == 1.).squeeze()
xb = pt.nonzero(pt.abs(X) == Lx / 2.).squeeze()
border = pt.cat([z0, z1, xb])
psi = 4. * Z * pt.tanh(X)
phi = mu * (1. - Z)
seed = pt.cat([psi, phi])
fz = 1. - Z ** 3
fzp = -3. * (Z ** 2)
# %% Loop
pt.cuda.synchronize()
tloop0 = time.time()
while do == 1 and k < kmax:
    b1 = -phi ** 2 * psi + fz * (Z * psi - (Dxx @ psi) - fzp * (Dz @ psi) - fz * (Dzz @ psi))
    b2 = 2. * phi * psi ** 2 - (Dxx @ phi) - fz * (Dzz @ phi)
    b1[z1] = fzp[z1] * (Dz[z1] @ psi) + (Dxx[z1] @ psi) - psi[z1]
    b1[z0] = psi[z0]
    b1[xb] = Dx[xb] @ psi
    b2[z1] = phi[z1]
    b2[z0] = phi[z0] - mu
    b2[xb] = Dx[xb] @ phi
    b = - pt.cat([b1, b2])
    A11 = pt.diag(Z * fz - phi ** 2) - (fz * Dxx.t()).t() - (fz ** 2 * Dzz.t()).t() - (fz * fzp * Dz.t()).t()
    A12 = pt.diag(-2. * phi * psi)
    A21 = pt.diag(4. * psi * phi)
    A22 = pt.diag(2. * psi ** 2) - Dxx - (fz * Dzz.t()).t()
    A11[z0] = E[z0]
    A11[z1] = (fzp[z1] * Dz[z1].t()).t() + Dxx[z1] - E[z1]
    A11[xb] = Dx[xb]
    A12[border] = A21[border] = 0.
    A22[z0] = E[z0]
    A22[z1] = E[z1]
    A22[xb] = Dx[xb]
    A = pt.cat([pt.cat([A11, A12], 1), pt.cat([A21, A22], 1)])
    pt.cuda.synchronize()
    bnt0 = time.time()
    sol = pt.solve(b.unsqueeze(1), A)[0]  # bottleneck
    print(pt.norm(b))
    pt.cuda.synchronize()
    bnt1 = time.time()
    #print("bottleneck time:", bnt1 - bnt0)
    seed = seed + sol.squeeze()
    psi, phi = pt.chunk(seed, 2)
    k = k + 1
    normsol = pt.norm(sol)
    if normsol < eps:
        do = 0
pt.cuda.synchronize()
tloop1 = time.time()
# %% Visualization
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(zz.cpu().numpy(), xx.cpu().numpy(), psi.cpu().numpy().reshape((Nz, Nx)), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
pt.cuda.synchronize()
t1 = time.time()
print("total time:", t1 - t0)
print("loop time:", tloop1 - tloop0)
