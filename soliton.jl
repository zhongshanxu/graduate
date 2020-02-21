using LinearAlgebra
using Plots

chebp(n) = cos.(pi*collect(0:n-1)/(n-1))
function cheb(m,l)
    x=chebp(m)
    c=ones(m)
    c[1] = c[end] = 2.
    d = [i==j ? 0 : (2/l)*c[i]/c[j]*(-1)^(i+j)/(x[i]-x[j]) for i = 1:m, j = 1:m]
    d-Diagonal(vec(sum(d,dims=2)))
end
function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where {T}
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repeat(vx, m, 1), repeat(vy, 1, n))
end

mu = 5.
Lz, Lx = 1., 24.
Nz, Nx = 30, 60
N = Nz * Nx
z, x = chebp(Nz), chebp(Nx)
z = (1 .+ z)/2
x = (Lx/2) * x
dz, dx = cheb(Nz,Lz), cheb(Nx,Lx)
zz, xx = meshgrid(z,x)
Z, X = zz[:], xx[:]
Ez = Matrix{Float64}(I,Nz,Nz)
Ex = Matrix{Float64}(I,Nx,Nx)
E = Matrix{Float64}(I,N,N)
Dz, Dx = kron(dz,Ex), kron(Ez,dx)
Dz2, Dx2 = kron(dz^2,Ex), kron(Ez,dx^2)
z0 = findall(in(0.),Z)
z1 = findall(in(1.),Z)
xb = findall(in(Lx/2),abs.(X))
border = union(z0,z1,xb)
psi = 4Z .* tanh.(X)
phi = mu*(1 .- Z)
seed = [psi;phi]
fz = 1 .- Z.^3
fzp = -3Z.^2
k, kmax, epsilon = 0, 10, 1e-6
done = true
while done && k < kmax
    # computing b
    b1 = -phi.^2 .* psi + fz.*(Z.*psi - Dx2*psi - fzp.*(Dz*psi) - fz.*(Dz2*psi))
    b2 = 2phi .* psi.^2 - Dx2*phi - fz.*(Dz2*phi)
    b1[z1] = fzp[z1].*(Dz[z1,:]*psi) + Dx2[z1,:]*psi - psi[z1]
    b1[z0] = psi[z0]
    b1[xb] = Dx[xb,:]*psi
    b2[z1] = phi[z1]
    b2[z0] = phi[z0] .- mu
    b2[xb] = Dx[xb,:]*phi
    b = -[b1;b2]
    # computing A
    A11 = Diagonal(Z.*fz - phi.^2) - fz.*Dx2 - fz.^2 .* Dz2 - fz.*fzp.*Dz
    A12 = Diagonal(-2phi.*psi)
    A21 = Diagonal(4psi.*phi)
    A22 = Diagonal(2psi.^2) - Dx2 - fz.*Dz2
    A11[z0,:] = E[z0,:]
    A11[z1,:] = fzp[z1].*Dz[z1,:] + Dx2[z1,:] - E[z1,:]
    A11[xb,:] = Dx[xb,:]
    A12[border,:] .= 0.
    A21[border,:] .= 0.
    A22[z0,:] = E[z0,:]
    A22[z1,:] = E[z1,:]
    A22[xb,:] = Dx[xb,:]
    A = [A11 A12;A21 A22]
    sol = A \ b
    global seed += sol
    reseed = reshape(seed,:,2)
    global psi = reseed[:,1]
    global phi = reseed[:,2]
    global k += 1
    rmssol = sqrt(sum(abs2.(sol))/(2N))
    if rmssol < epsilon
        global done = false
    end
    println(rmssol)
end
plot(z,x,reshape(psi,Nx,Nz),st=:surface)
