function [ c_return ] = pdf_MF_normal_expansion( s, bool_scaled, tol )
% evaluate the normalizing constant of matrix Fisher distribution using
% series expansion.
% ALERT: s MUST be small to avoid numerical overflow!

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

if nargin < 2
    bool_scaled = false;
end

if nargin < 3
    tol = 1e-6;
end

% convert to Bingham distribution
S = diag(s);
B = [2*S-trace(S)*eye(3),zeros(3,1);zeros(1,3),trace(S)];
lambda = diag(B);
lambdaScale = lambda(4);
lambda = lambda-lambdaScale;

% calculate multiplicity
phi = unique(lambda);
d = sum(repmat(phi',4,1)==lambda)';

% calculate number of iterations based on required accuracy
options = optimoptions('fsolve','display','off');
norm1phi = sum(abs(phi));
err = @(N) prod(repmat(norm1phi,1,floor(N))./(floor(N):-1:1))*(N+1)/(N+1-norm1phi)*2*pi^2;
N = floor(fsolve(@(N) err(N)-tol,norm1phi,options));

% iteration
q = length(phi);
c = 0;
for n = 0:N
    perm = getPermute(n,q);
    
    Np = size(perm,2);
    for np = 1:Np
        c = c+calc(phi,d,q,perm(:,np));
    end
end

c_return = c/prod(gamma(d/2));

if ~bool_scaled
    c_return = c_return*exp(lambdaScale);
end

end


function perm = getPermute(N,q)

perm = [];
if q>1
    for n = 0:N
        subperm = getPermute(N-n,q-1);
        col = size(subperm,2);
        perm = [perm,[repmat(n,1,col);subperm]];
    end
else
    perm = N;
end

end


function result = calc(phi,d,q,perm)

result = 1;
for i = 1:q
    if perm(i)~=0
        result = result*prod(repmat(phi(i),1,perm(i))./(perm(i):-1:1));
    end
    result = result*gamma(perm(i)+d(i)/2);
end

result = result/gamma(sum(perm)+sum(d)/2);

end

