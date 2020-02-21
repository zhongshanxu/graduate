mu=5.0;lz=1.0;lx=24.0;
Nz=39;Nx=69;N=(Nz+1)*(Nx+1);
[dz,z]=cheb(Nz);dz=(2/lz)*dz;z=(1+z)/2;
[dx,x]=cheb(Nx);dx=(2/lx)*dx;x=(lx/2)*x;
Iz=eye(Nz+1);Ix=eye(Nx+1);I=eye(N);
Dz=kron(dz,Ix);Dx=kron(Iz,dx);
Dzz=kron(dz^2,Ix);Dxx=kron(Iz,dx^2);
[zz,xx]=meshgrid(z,x);
Z=zz(:);X=xx(:);
z0=find(Z==0);
z1=find(Z==1);
xb=find(abs(X)==lx/2);
border=find(Z==0|Z==1|abs(X)==lx/2);
psi=4*Z.*tanh(X);phi=mu*(1-Z);seed=[psi;phi];%initialization
fz=1-Z.^3;fpz=-3*Z.^2;
k=0;kmax=66;eps=1e-6;sol=seed;
tic
while norm(sol,'inf')>eps && k<kmax
%computing b
b1=-phi.^2.*psi+fz.*(Z.*psi-(Dxx*psi)-fpz.*(Dz*psi)-fz.*(Dzz*psi));
b2=2*phi.*psi.^2-(Dxx*phi)-fz.*(Dzz*phi);
b1(z1)=fpz(z1).*(Dz(z1,:)*psi)+(Dxx(z1,:)*psi)-psi(z1);
b1(z0)=psi(z0);
b1(xb)=Dx(xb,:)*psi;
b2(z1)=phi(z1);
b2(z0)=phi(z0)-mu;
b2(xb)=Dx(xb,:)*phi;
b=-[b1;b2];
%computing A
A11=diag(Z.*fz-phi.^2)-fz.*Dxx-fz.^2.*Dzz-fz.*fpz.*Dz;
A12=diag(-2*phi.*psi);
A21=diag(4*psi.*phi);
A22=diag(2*psi.^2)-Dxx-fz.*Dzz;
A11(z0,:)=I(z0,:);A11(z1,:)=fpz(z1).*Dz(z1,:)+Dxx(z1,:)-I(z1,:);
A11(xb,:)=Dx(xb,:);
A12(border,:)=0;
A21(border,:)=0;
A22(z0,:)=I(z0,:);A22(z1,:)=I(z1,:);
A22(xb,:)=Dx(xb,:);
A=[A11,A12;A21,A22];
sol=A\b;%solve linear systems
seed=sol+seed;
seed=reshape(seed,[],2);
psi=seed(:,1);
phi=seed(:,2);
seed=seed(:);
k=k+1;
end
toc
mesh(zz,xx,reshape(psi,[Nx+1,Nz+1]))

  function [D,x] = cheb(N)
  if N==0, D=0; x=1; return, end
  x = cos(pi*(0:N)/N)'; 
  c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
  X = repmat(x,1,N+1);
  dX = X-X';                  
  D  = (c*(1./c)')./(dX+(eye(N+1)));      % off-diagonal entries
  D  = D - diag(sum(D'));                 % diagonal entries
  end
