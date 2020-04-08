function [ c_return ] = pdf_MF_normal_saddle( s, bool_scaled )
% evaluate the normalizing constant of matrix Fisher distribution using
% saddle point approximations

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

% if bool_scaled is not defined, then set it false
if nargin < 2
    bool_scaled=false;
end

% convert to Bingham distribution
S = diag(s);
B = [2*S-trace(S)*eye(3),zeros(3,1);zeros(1,3),trace(S)];
lambda = -diag(B);

% scaled or unscaled
if bool_scaled
    lambda = lambda-lambda(4);
end

% calculate t0
dKtCoeff = [2, 4-2*sum(lambda),...
    2*sum(prod(lambda(combnk(1:4,2)),2))-3*sum(lambda),...
    -2*sum(prod(lambda(combnk(1:4,3)),2))+2*sum(prod(lambda(combnk(1:4,2)),2)),...
    2*prod(lambda)-sum(prod(lambda(combnk(1:4,3)),2))];
t0 = roots(dKtCoeff);
t0 = t0(t0 <= min(lambda));

% first order saddle point approximation
ddKt0 = 0.5*sum(1./(lambda-t0).^2);
c1 = 2^(1/2)*pi^(3/2)*ddKt0^(-1/2)*prod((lambda-t0).^(-1/2))*exp(-t0);

% second order saddle point approximation
d3Kt0 = sum(1./(lambda-t0).^3);
d4Kt0 = 3*sum(1./(lambda-t0).^4);

rho3 = d3Kt0/ddKt0^(3/2);
rho4 = d4Kt0/ddKt0^2;
T = 1/8*rho4-5/24*rho3^2;

c3 = c1*exp(T);

c_return = c3/(2*pi^2);

end

