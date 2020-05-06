function [ c_return, dc_return, ddc_return ] = pdf_MF_normal_saddle( s, bool_scaled, bool_dc, bool_ddc )
% evaluate the normalizing constant of matrix Fisher distribution using
% saddle point approximations

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

% if bool_scaled is not defined, then set it false
if nargin < 2
    bool_scaled=false;
end
if nargin < 3
    bool_dc=false;
end
if nargin < 4
    bool_ddc=false;
end
if bool_ddc
    bool_dc=true;
end

% convert to Bingham distribution
S = diag(s);
B = [2*S-trace(S)*eye(3),zeros(3,1);zeros(1,3),trace(S)];
lambda = -diag(B)';

% scaled or unscaled
if bool_scaled
    lambda = lambda-lambda(4);
end

%% normalizing constant
% calculate t0
dKtCoeff = [2, 4-2*sum(lambda),...
    2*sum(prod(lambda(combnk(1:4,2)),2))-3*sum(lambda),...
    -2*sum(prod(lambda(combnk(1:4,3)),2))+2*sum(prod(lambda(combnk(1:4,2)),2)),...
    2*prod(lambda)-sum(prod(lambda(combnk(1:4,3)),2))];
t0 = roots(dKtCoeff);
t0(imag(t0)~=0)=[];
t0 = min(t0);

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

if bool_scaled
    c_bar = c3/(2*pi^2);
    c_return = c_bar;
else
    c = c3/(2*pi^2);
    c_return = c;
end

if ~bool_dc
    return;
end

%% first order derivative
% calculate multiplicity
phi = unique(lambda);
m = sum(repmat(phi',1,4)==lambda,2);

dc = zeros(4,0);
for i = 1:3
    lambdad = [lambda(i),lambda(i),lambda];
    
    % calculate multiplicity
    phi = unique(lambdad);
    d = sum(repmat(phi',1,6)==lambdad,2);
    
    % calculate t0
    dKtCoeff = [2, 6-2*sum(lambdad),...
        2*sum(prod(lambdad(combnk(1:6,2)),2))-5*sum(lambdad),...
        -2*sum(prod(lambdad(combnk(1:6,3)),2))+4*sum(prod(lambdad(combnk(1:6,2)),2)),...
        2*sum(prod(lambdad(combnk(1:6,4)),2))-3*sum(prod(lambdad(combnk(1:6,3)),2)),...
        -2*sum(prod(lambdad(combnk(1:6,5)),2))+2*sum(prod(lambdad(combnk(1:6,4)),2)),...
        2*prod(lambdad)-sum(prod(lambdad(combnk(1:6,5)),2))];
    t0 = roots(dKtCoeff);
    t0(imag(t0)~=0)=[];
    t0 = min(t0);
    
    % first order saddle point approximation
    ddKt0 = 0.5*sum(1./(lambdad-t0).^2);
    c1 = 2^(1/2)*pi^(5/2)*ddKt0^(-1/2)*prod((lambdad-t0).^(-1/2))*exp(-t0);

    % second order saddle point approximation
    d3Kt0 = sum(1./(lambdad-t0).^3);
    d4Kt0 = 3*sum(1./(lambdad-t0).^4);

    rho3 = d3Kt0/ddKt0^(3/2);
    rho4 = d4Kt0/ddKt0^2;
    T = 1/8*rho4-5/24*rho3^2;

    c3 = c1*exp(T);
    
    dc(i) = c3/(2*pi^2);
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
                if abs(s(i))~=abs(s(k(1))) && abs(s(i))~=abs(s(k(2)))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2)-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                elseif abs(s(i))~=abs(s(k(1))) && abs(s(i))==abs(s(k(2))) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2*sign(s(i)*s(k(2)));
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2)-dc(i)/2/s(i);
                elseif abs(s(i))~=abs(s(k(1))) && s(i)==s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(1))*s(k(1)))/(s(k(1))^2-s(i)^2);
                elseif abs(s(i))==abs(s(k(1))) && abs(s(i))~=abs(s(k(2))) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2*sign(s(i)*s(k(1)));
                    b(3*(i-1)+j) = c-dc(i)/2/s(i)-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                elseif abs(s(i))==abs(s(k(1))) && abs(s(i))==abs(s(k(2))) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2*sign(s(i)*s(k(1)));
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2*sign(s(i)*s(k(2)));
                    b(3*(i-1)+j) = c-dc(i)/s(i);
                elseif s(i)==s(k(1)) && abs(s(i))~=abs(s(k(2))) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = c-(-dc(i)*s(i)+dc(k(2))*s(k(2)))/(s(k(2))^2-s(i)^2);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 3;
                    b(3*(i-1)+j)=c;
                end
            else
                if abs(s(i))~=abs(s(j))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = dc(k)+(-dc(i)*s(j)+dc(j)*s(i))/(s(j)^2-s(i)^2);
                elseif abs(s(i))==abs(s(j)) && s(i)~=0
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+i) = -1/2*sign(s(i)*s(j));
                    b(3*(i-1)+j) = dc(k)-dc(j)/2/s(i);
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
                if abs(s(i))~=abs(s(k(1))) && abs(s(i))~=abs(s(k(2)))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                elseif abs(s(i))~=abs(s(k(1))) && abs(s(i))==abs(s(k(2))) && s(i)~=0
                    sig = sign(s(i)*s(k(2)));
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2*sig;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - (1/2-sig/2+1/2/s(i))*c_bar-(1/2/s(i)+1-sig/2)*dc_bar(i)+sig*dc_bar(k(2))/2;
                elseif abs(s(i))~=abs(s(k(1))) && s(i)==s(k(2)) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar/(s(i)+s(k(1)))+dc_bar(i)*s(i)/(s(k(1))^2-s(i)^2)-dc_bar(k(1))*s(k(1))/(s(k(1))^2-s(i)^2)...
                        - c_bar-2*dc_bar(i);
                elseif abs(s(i))==abs(s(k(1))) && abs(s(i))~=abs(s(k(2))) && s(i)~=0
                    sig = sign(s(i)*s(k(1)));
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2*sign(s(i)*s(k(1)));
                    b(3*(i-1)+j) = -2*dc_bar(i) - (1/2-sig/2+1/2/s(i))*c_bar-(1/2/s(i)+1-sig/2)*dc_bar(i)+sig*dc_bar(k(1))/2 ...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                elseif abs(s(i))==abs(s(k(1))) && abs(s(i))==abs(s(k(2))) && s(i)~=0
                    sig1 = sign(s(i)*s(k(1)));
                    sig2 = sign(s(i)*s(k(2)));
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    A(3*(i-1)+j,3*(i-1)+k(1)) = -1/2*sig1;
                    A(3*(i-1)+j,3*(i-1)+k(2)) = -1/2*sig2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - (1/2-sig1/2+1/2/s(i))*c_bar-(1/2/s(i)+1-sig1/2)*dc_bar(i)+sig1*dc_bar(k(1))/2 ...
                        - (1/2-sig2/2+1/2/s(i))*c_bar-(1/2/s(i)+1-sig2/2)*dc_bar(i)+sig2*dc_bar(k(2))/2;
                elseif s(i)==s(k(1)) && abs(s(i))~=abs(s(k(2))) && s(i)==0
                    A(3*(i-1)+j,3*(i-1)+j) = 2;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar-2*dc_bar(i)...
                        - c_bar/(s(i)+s(k(2)))+dc_bar(i)*s(i)/(s(k(2))^2-s(i)^2)-dc_bar(k(2))*s(k(2))/(s(k(2))^2-s(i)^2);
                else
                    A(3*(i-1)+j,3*(i-1)+j) = 3;
                    b(3*(i-1)+j) = -2*dc_bar(i) - c_bar-2*dc_bar(i) - c_bar-2*dc_bar(i);
                end
            else
                if abs(s(i))~=abs(s(j))
                    A(3*(i-1)+j,3*(i-1)+j) = 1;
                    b(3*(i-1)+j) = -c_bar/(s(i)+s(j)) - (1+s(j)/(s(j)^2-s(i)^2))*dc_bar(i)...
                        - (1-s(i)/(s(j)^2-s(i)^2))*dc_bar(j) + dc_bar(k);
                elseif abs(s(i))==abs(s(j)) && s(i)~=0
                    sig = sign(s(i)*s(j));
                    A(3*(i-1)+j,3*(i-1)+j) = 3/2;
                    A(3*(i-1)+j,3*(i-1)+i) = -1/2*sig;
                    b(3*(i-1)+j) = -(1/2-sig/2+1/2/s(i))*c_bar - (3/2-sig)*dc_bar(i) - (3/2+1/2/s(i))*dc_bar(j) + dc_bar(k);
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

