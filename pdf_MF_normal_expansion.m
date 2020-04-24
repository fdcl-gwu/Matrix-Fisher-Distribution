function [ c_return, dc_return, ddc_return ] = pdf_MF_normal_expansion( s, bool_scaled, bool_dc, bool_ddc, tol )
% evaluate the normalizing constant of matrix Fisher distribution using
% series expansion.
% ALERT: s MUST be small to avoid numerical overflow!

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

if nargin < 2
    bool_scaled = false;
end
if nargin < 3
    bool_dc = false;
end
if nargin < 4
    bool_ddc = false;
end
if nargin < 5
    tol = 1e-6;
end
if bool_ddc
    bool_dc = true;
end

% convert to Bingham distribution
S = diag(s);
B = [2*S-trace(S)*eye(3),zeros(3,1);zeros(1,3),trace(S)];
lambda = diag(B)';
if bool_scaled
    lambda = lambda-lambda(4);
end

%% normalizing constant
% calculate multiplicity
phi = unique(lambda);
m = sum(repmat(phi',1,4)==lambda,2);

% calculate number of iterations based on required accuracy
options = optimoptions('fsolve','display','off');
norm1phi = sum(abs(phi));
err = @(N) prod(repmat(norm1phi,1,floor(N))./(floor(N):-1:1))*(N+1)/(N+1-norm1phi)*2*pi^2;
N = floor(fsolve(@(N) err(N)-tol,norm1phi,options));

% iteration
q = length(phi);
c = 0;
cOld = nan;
for n = 0:N
    perm = getPermute(n,q);
    
    Np = size(perm,2);
    for np = 1:Np
        c = c+calc(phi,m,q,perm(:,np));
    end
    
    if c==cOld
        break;
    end
    cOld = c;
end

if ~bool_scaled
    c = c/prod(gamma(m/2));
    c_return = c;
else
    c_bar = c/prod(gamma(m/2));
    c_return = c_bar;
end

if ~bool_dc
    return;
end

%% first order derivative
dc = zeros(4,1);
dcOld = nan(4,1);
for i = 1:4
    % reduce computation cost by ultilizing multiplicity
    if i>1 && lambda(i)==lambda(i-1)
        dc(i) = dc(i-1);
        continue;
    end
    
    % calculate derivatives using Bingham distribution in higher
    % dimensional sphere
    lambdad = [lambda(i),lambda(i),lambda];
    
    % calculate multiplicity
    phi = unique(lambdad);
    d = sum(repmat(phi',1,6)==lambdad,2);

    % calculate number of iterations based on required accuracy
    options = optimoptions('fsolve','display','off');
    norm1phi = sum(abs(phi));
    err = @(N) prod(repmat(norm1phi,1,floor(N))./(floor(N):-1:1))*(N+1)/(N+1-norm1phi)*2*pi^2;
    N = floor(fsolve(@(N) err(N)-tol,norm1phi,options));

    % iteration
    q = length(phi);
    for n = 0:N
        perm = getPermute(n,q);

        Np = size(perm,2);
        for np = 1:Np
            dc(i) = dc(i)+calc(phi,d,q,perm(:,np));
        end
        
        if dc(i)==dcOld(i)
            break;
        end
        dcOld(i)=dc(i);
    end
    
    dc(i) = dc(i)/prod(gamma(d/2))*pi;
    dc(i) = dc(i)*pi^-1*prod(gamma(d/2)./gamma(m/2));
    dc(i) = dc(i)/sum(lambda(i)==lambda);
end

if ~bool_scaled
    dc(4) = c-dc(1)-dc(2)-dc(3);
    dc = [dc(1)-dc(2)-dc(3)+dc(4),...
        dc(2)-dc(1)-dc(3)+dc(4),...
        dc(3)-dc(1)-dc(2)+dc(4)];
    dc_return = dc;
else
    dc = -2*[dc(2)+dc(3),dc(1)+dc(3),dc(1)+dc(2)];
    dc_bar = dc;
    dc_return = dc_bar;
end

if ~bool_ddc
    return;
end

%% second order derivatives
if ~bool_scaled
    A = zeros(9,9);
    b = zeros(9,1);

    for i = 1:3
        for j = 1:3
            k = setdiff(1:3,[i,j]);
            if i==j
                if s(i)~=s(k(1)) && s(i)~=s(k(2))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2)-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                elseif s(i)~=s(k(1)) && s(i)==s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2)-dc(i)/2/s(i);
                elseif s(i)~=s(k(1)) && s(i)==s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2);
                elseif s(i)==s(k(1)) && s(i)~=s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2;
                    b(3*(i-1)+j) = c-dc(i)/2/s(i)-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                elseif s(i)==s(k(1)) && s(i)==s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2;
                    b(3*(i-1)+j) = c-dc(i)/s(i);
                elseif s(i)==s(k(1)) && s(i)~=s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 3;
                    b(3*(i-1)+j)=c;
                end
            else
                if s(i)~=s(j)
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = dc(k)+(-dc(i)*s(j)+dc(j)*s(i))/(s(j)^2-s(i)^2);
                elseif s(i)==s(j) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+i) = -1/2;
                    b(3*(i-1)+j) = dc(k)-dc(i)/2/s(i);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = dc(k);
                end
            end
        end
    end

    ddc = A\b;
    ddc = [ddc(1:3),ddc(4:6),ddc(7:9)];
    ddc_return = ddc;
else
    A = zeros(9,9);
    b = zeros(9,1);

    for i = 1:3
        for j = 1:3
            k = setdiff(1:3,[i,j]);
            if i==j
                if s(i)~=s(k(1)) && s(i)~=s(k(2))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                elseif s(i)~=s(k(1)) && s(i)==s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - c_bar/2/s(i)-(1/2/s(i)+1/2)*dc_bar(i)+dc_bar(k(2))/2;
                elseif s(i)~=s(k(1)) && s(i)==s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - c_bar-2*dc_bar(i);
                elseif s(i)==s(k(1)) && s(i)~=s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/2/s(i)-(1/2/s(i)+1/2)*dc_bar(i)+dc_bar(k(1))/2 ...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                elseif s(i)==s(k(1)) && s(i)==s(k(2)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/2/s(i)-(1/2/s(i)+1/2)*dc_bar(i)+dc_bar(k(1))/2 ...
                        - c_bar/2/s(i)-(1/2/s(i)+1/2)*dc_bar(i)+dc_bar(k(2))/2;
                elseif s(i)==s(k(1)) && s(i)~=s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar-2*dc_bar(i)...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 3;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar-2*dc_bar(i) - c_bar-2*dc_bar(i);
                end
            else
                if s(i)~=s(j)
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = -c_bar/(s(i)+s(j)) - (1+s(j)/(s(j)^2-s(i)^2))*dc_bar(i)...
                        - (1-s(i)/(s(j)^2-s(i)^2))*dc_bar(j) + dc_bar(k);
                elseif s(i)==s(j) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+i) = -1/2;
                    b(3*(i-1)+j) = -c_bar/2/s(i) - (1/2+1/2/s(i))*dc_bar(i) - 3/2*dc_bar(j) + dc_bar(k);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = -c_bar - 2*dc_bar(i) - 2*dc_bar(j) + dc_bar(k);
                end
            end
        end
    end

    ddc_bar = A\b;
    ddc_bar = [ddc_bar(1:3),ddc_bar(4:6),ddc_bar(7:9)];
    ddc_return = ddc_bar;
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

