function [ c_return, dc_return, ddc_return ] = pdf_MF_normal_holonomic( s, bool_scaled, bool_dc, bool_ddc )
% evaluating the normalising constant of matrix Fisher distribution using
% the holonomic method

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

% if bool_scaled is not defined, then set it false
if nargin < 2
    bool_scaled=false;
end
if nargin < 3
    bool_dc = false;
end
if nargin < 4
    bool_ddc = false;
end

%% normalizing constant and its first order derivative
% initial value
s0 = [1/8,1/16,0];
if ~bool_scaled
    g0 = [1.003259407399006,0.041764420609303,0.020906658448392,0.001304628523573];
else
    g0 = [0.831731261822189,-0.797107341033143,-0.814399033204620,-0.830649686787739];
end

% ODE
if ~bool_scaled
    [~,g] = ode45(@(t,g) ODE(t,g,s,s0),[0,1],g0);
else
    [~,g] = ode45(@(t,g) ODE_scaled(t,g,s,s0),[0,1],g0);
end

c_return = g(end,1);
dc_return = g(end,2:4);

if ~bool_ddc
    return;
end

%% second order derivative
if ~bool_scaled
    c = g(end,1);
    dc = g(end,2:4);
else
    c_bar = g(end,1);
    dc_bar = g(end,2:4);
end

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


function P = pfaffian( s )

P(:,:,1) = [0,1,0,0;
    1,-s(1)/(s(1)^2-s(2)^2)-s(1)/(s(1)^2-s(3)^2),s(2)/(s(1)^2-s(2)^2),s(3)/(s(1)^2-s(3)^2);
    0,s(2)/(s(1)^2-s(2)^2),-s(1)/(s(1)^2-s(2)^2),1;
    0,s(3)/(s(1)^2-s(3)^2),1,-s(1)/(s(1)^2-s(3)^2)];

P(:,:,2) = [0,0,1,0;
    0,-s(2)/(s(2)^2-s(1)^2),s(1)/(s(2)^2-s(1)^2),1;
    1,s(1)/(s(2)^2-s(1)^2),-s(2)/(s(2)^2-s(1)^2)-s(2)/(s(2)^2-s(3)^2),s(3)/(s(2)^2-s(3)^2);
    0,1,s(3)/(s(2)^2-s(3)^2),-s(2)/(s(2)^2-s(3)^2)];

P(:,:,3) = [0,0,0,1;
    0,-s(3)/(s(3)^2-s(1)^2),1,s(1)/(s(3)^2-s(1)^2);
    0,1,-s(3)/(s(3)^2-s(2)^2),s(2)/(s(3)^2-s(2)^2);
    1,s(1)/(s(3)^2-s(1)^2),s(2)/(s(3)^2-s(2)^2),-s(3)/(s(3)^2-s(1)^2)-s(3)/(s(3)^2-s(2)^2)];

end


function P = pfaffian_scaled( s )

P(:,:,1) = [0,1,0,0;
    -1/(s(1)+s(2))-1/(s(1)+s(3)),-2-s(1)/(s(1)^2-s(2)^2)-s(1)/(s(1)^2-s(3)^2),s(2)/(s(1)^2-s(2)^2),s(3)/(s(1)^2-s(3)^2);
    -1/(s(1)+s(2)),-1+s(2)/(s(1)^2-s(2)^2),-1-s(1)/(s(1)^2-s(2)^2),1;
    -1/(s(1)+s(3)),-1+s(3)/(s(1)^2-s(3)^2),1,-1-s(1)/(s(1)^2-s(3)^2)];

P(:,:,2) = [0,0,1,0;
    -1/(s(2)+s(1)),-1-s(2)/(s(2)^2-s(1)^2),-1+s(1)/(s(2)^2-s(1)^2),1;
    -1/(s(2)+s(1))-1/(s(2)+s(3)),s(1)/(s(2)^2-s(1)^2),-2-s(2)/(s(2)^2-s(1)^2)-s(2)/(s(2)^2-s(3)^2),s(3)/(s(2)^2-s(3)^2);
    -1/(s(2)+s(3)),1,-1+s(3)/(s(2)^2-s(3)^2),-1-s(2)/(s(2)^2-s(3)^2)];

P(:,:,3) = [0,0,0,1;
    -1/(s(3)+s(1)),-1-s(3)/(s(3)^2-s(1)^2),1,-1+s(1)/(s(3)^2-s(1)^2);
    -1/(s(3)+s(2)),1,-1-s(3)/(s(3)^2-s(2)^2),-1+s(2)/(s(3)^2-s(2)^2);
    -1/(s(3)+s(1))-1/(s(3)+s(2)),s(1)/(s(3)^2-s(1)^2),s(2)/(s(3)^2-s(2)^2),-2-s(3)/(s(3)^2-s(1)^2)-s(3)/(s(3)^2-s(2)^2)];

end


function f = ODE( t, g, s, s0 )

st = (1-t)*s0+t*s;
P = pfaffian(st);

f = 0;
for i = 1:3
    f = f + P(:,:,i)*(s(i)-s0(i))*g;
end

end


function f = ODE_scaled( t, g, s, s0 )

st = (1-t)*s0+t*s;
P = pfaffian_scaled(st);

f = 0;
for i = 1:3
    f = f + P(:,:,i)*(s(i)-s0(i))*g;
end

end

