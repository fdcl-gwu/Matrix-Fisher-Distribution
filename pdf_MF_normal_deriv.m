function varargout=pdf_MF_normal_deriv(s,bool_ddc,bool_scaled)
%pdf_MF_norma_deriv: the derivatives of the normalizing constant for the matrix Fisher distribution
%on SO(3)
%   [dc, ddc] = pdf_MF_normal(s,BOOL_DDC,BOOL_SCALED) returns the 3x1 first
%   order derivative dc and the 3x3 second order derivatives ddc of the normalizing
%   constant with respect to the proper singular values for the matrix Fisher
%   distribution on SO(3), for a given 3x1 (or 1x3) proper singular
%   values s.
%
%   BOOL_DDC determines whether the second order derivative
%   are computed or not:
%       0 - (defalut) is the same as dc=pdf_MF_normal_deriv(s), and the second
%       order derivatives are not computed
%       1 - computes the second order derivatives, and reuturns ddc
%
%   BOOL_SCALED determines whether the normalizing constant is
%   scaled or not:
%       0 - (defalut) is the same as pdf_MF_normal_deriv(s,BOOL_DDC), and
%       then derivatives of the unscaled normalizing constant c are returned
%       1 - computes the derivatives of the exponentially scaled normalizing constant,
%       c_bar = exp(-sum(s))*c
%
%   Examples
%       dc=pdf_MF_normal_deriv(s) - first order derivatives of c
%       [dc, ddc]=pdf_MF_normal_deriv(s,true) - first and second
%       order derivatives of c
%
%       dc_bar=pdf_MF_normal_deriv(s,false,true) - first order
%       derivatives of the exponentially scaled c
%       [dc_bar, ddc_bar]=pdf_MF_normal_deriv(s,true,true) - first and second
%       order derivatives of the exponentially scaled c
%
%   See T. Lee, "Bayesian Attitude Estimation with the Matrix Fisher
%   Distribution on SO(3)", 2017, http://arxiv.org/abs/1710.03746
%
%   See also PDF_MF_NORMAL_DERIV_APPROX

if nargin < 3
    bool_scaled = false;
end
if nargin < 2
    bool_ddc = false;
end

if ~bool_scaled
    dc=zeros(3,1);
    
    % derivatives of the normalizing constant
    for i=1:3
        dc(i) = integral(@(u) f_kunze_s_deriv_i(u,s,i),-1,1);
    end
    varargout{1}=dc;
    
    if bool_ddc
        % compute the second order derivatives of the normalizing constant
        c = pdf_MF_normal(s);
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
        
        varargout{2}=ddc;
    end
    
else
    % derivatives of the scaled normalizing constant
    dc_bar = zeros(3,1);
    
    for i=1:3
        index=circshift([1 2 3],[0 4-i]);
        j=index(2);
        k=index(3);
        
        dc_bar(k) = integral(@(u) f_kunze_s_deriv_scaled(u,[s(i),s(j),s(k)]),-1,1);
    end
    varargout{1}=dc_bar;
    
    % compute the second order derivatives of the scaled normalizing
    % constant
    if bool_ddc
        c_bar = pdf_MF_normal(s,1);
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
        
        varargout{2}=ddc_bar;
    end
end
end

function Y=f_kunze_s_deriv_scaled(u,s)
% integrand for the derivative of the scaled normalizing constant

J=besseli(0,1/2*(s(1)-s(2))*(1-u),1).*besseli(0,1/2*(s(1)+s(2))*(1+u),1);
Y=1/2*J.*(u-1).*exp((min([s(1) s(2)])+s(3))*(u-1));

end


function Y=f_kunze_s_deriv_i(u,s,i)
% integrand for the derivative of the normalizing constant
index=circshift([1 2 3],[0 4-i]);
j=index(2);
k=index(3);

J00=besseli(0,1/2*(s(j)-s(k))*(1-u)).*besseli(0,1/2*(s(j)+s(k))*(1+u));
Y=1/2*J00.*u.*exp(s(i)*u);

end
