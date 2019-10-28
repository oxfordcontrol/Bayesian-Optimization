k = 3;
I = eye(k);

%A = [I I I;
%    I 0*I 0*I
%    0*I I 0*I
%    0*I 0*I I]';

A = kron(ones(1, k), I);
n = size(A, 1);
m = size(A, 2);

sigma = 1e-6;
rho = 0.1;
K = [sigma*eye(n) A; A' -eye(m)/rho];
a = randn(n, 1);
b = randn(m, 1);

z = K\[a; b];
x = z(1:n); y = z(n+1:n+m);

kron(ones(1, k), I)*kron(ones(k, 1), I)

x./(a + rho*A*b) - 1/(sigma + rho*n)