function [ c_return ] = pdf_MF_normal_holonomic( s, bool_scaled )
% evaluating the normalising constant of matrix Fisher distribution using
% the holonomic method

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

% if bool_scaled is not defined, then set it false
if nargin < 2
    bool_scaled=false;
end

% convert to Bingham distribution
S = diag(s);
B = [2*S-trace(S)*eye(3),zeros(3,1);zeros(1,3),trace(S)];
lambda = diag(B);
lambdaScale = lambda(4);
lambda = sort(lambda-lambdaScale);

% calculate multiplicity
phi = unique(lambda);
d = sum(repmat(phi',4,1)==lambda)';
q = length(d);

% initial value
if q==4
    phi0 = [-3/8;-2/8;-1/8;0];
    g0 = [0.831731261822204;0.195214171862506;0.203339498654065;0.211985344739804];
elseif q==3
    phi0 = [-2/8;-1/8;0];
    if d(1)==2
        g0 = [0.856890854121296;0.207496937135084*2;0.216275333628920];
    else
        g0 = [0.883646660593647;0.211851959219676;0.220767917330410*2];
    end
elseif q==2
    phi0 = [-1/8;0];
    g0 = [0.910954793294271;0.225330576237677*3];
else
    c_return = 1;
    return;
end

% ODE
[~,g] = ode45(@(t,g) ODE(t,g,phi,d,q,phi0),[0,1],g0);

c_return = g(end,1);

if ~bool_scaled
    c_return = c_return*exp(lambdaScale);
end

end


function P = pfaffian( phi, d, q )

P = zeros(q,q,q-1);
for i = 1:q-1
    P(1,:,i) = zeros(1,q);
    P(1,i+1,i) = 1;
    
    index = setdiff(1:q-1,i);
    P(i+1,1,i) = d(i)/2/phi(i);
    P(i+1,i+1,i) = 1-sum(d(index)/2./(phi(i)-phi(index)))-d(q)/2/phi(i)-d(i)/2/phi(i);
    for k = index
        P(i+1,k+1,i) = d(i)/2/(phi(i)-phi(k))-d(i)/2/phi(i);
    end
    
    for j = setdiff(1:q-1,i)
        P(j+1,:,i) = zeros(1,q);
        P(j+1,i+1,i) = d(j)/2/(phi(i)-phi(j));
        P(j+1,j+1,i) = -d(i)/2/(phi(i)-phi(j));
    end
end

end


function f = ODE( t, g, phi, d, q, phi0 )

phit = (1-t)*phi0+t*phi;
P = pfaffian(phit,d,q);

f = 0;
for i = 1:q-1
    f = f + P(:,:,i)*(phi(i)-phi0(i))*g;
end

end

