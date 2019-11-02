%% Dual Formulation
k = 3;
I = eye(k*(k + 1)/2);

A1t = kron(ones(k, 1), I);
n = size(A1t, 1);
m = size(A1t, 2);
A2t = eye(n);
At = [A1t A2t];


sigma = 1e-6;
rho1 = 10;
rho2 = 0.1;
K = [sigma*eye(n) At; At' -blkdiag(eye(m)/rho1, eye(n)/rho2)];
a = randn(n, 1);
b1 = randn(m, 1);
b2 = randn(n, 1);

z = K\[a; b1; b2];
x = z(1:n); y1 = z(n+1:n+m); y2 = z(n+m+1:end);

y2 - (rho2*x - rho2*b2)
y1 - (rho1*A1t'*x - b1*rho1)
(rho2 + sigma)*x + A1t*y1 - (a + b2*rho2)

y1_ = (A1t'*(a + b2*rho2)/(rho2 + sigma) - b1)./(1/rho1 + k/(rho2 + sigma));
x_ = (- A1t*y1_ + a + b2*rho2)/(rho2 + sigma);
y2_ = rho2* (x_ - b2);

norm(z - [x_; y1_; y2_])


%% Primal Formulation
k = 3;
I = eye(k);

At = kron(ones(1, k), I);
n = size(At, 1);
m = size(At, 2);

sigma = 1e-6;
rho = 0.1;
K = [sigma*eye(n) At; At' -eye(m)/rho];
a = randn(n, 1);
b = randn(m, 1);

z = K\[a; b];
x = z(1:n); y = z(n+1:n+m);

kron(ones(1, k), I)*kron(ones(k, 1), I)

x./(a + rho*At*b) - 1/(sigma + rho*n)