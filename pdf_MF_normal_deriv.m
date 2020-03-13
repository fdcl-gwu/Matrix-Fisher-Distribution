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
        ddc=zeros(3,3);
        
        if length(unique(s))==3
            % if s are different, use the first order derivative to
            % calculate the second order derivative
            c = pdf_MF_normal(s);
            for i=1:3
                jk = setdiff(1:3,i);
                j = jk(1);
                k = jk(2);
                ddcdTij = -dc(i)*s(i)/(s(j)^2-s(i)^2) + dc(j)*s(j)/(s(j)^2-s(i)^2);
                ddcdTik = -dc(i)*s(i)/(s(k)^2-s(i)^2) + dc(k)*s(k)/(s(k)^2-s(i)^2);
                ddc(i,i) = c-ddcdTij-ddcdTik;
                for j=i+1:3
                    k = setdiff(1:3,[i,j]);
                    dcdTkk = dc(k);
                    ddcdTij = -dc(i)*s(j)/(s(j)^2-s(i)^2) + dc(j)*s(i)/(s(j)^2-s(i)^2);
                    ddc(i,j) = dcdTkk+ddcdTij;
                    ddc(j,i) = ddc(i,j);
                end
            end
        else
            for i=1:3
                ddc(i,i) = integral(@(u) f_kunze_s_deriv_ii(u,s,i),-1,1);
                for j=i+1:3
                    ddc(i,j) = integral(@(u) f_kunze_s_deriv_ij(u,s,i,j),-1,1);
                    ddc(j,i)=ddc(i,j);
                end
            end
        end
        
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
        ddc_bar=zeros(3,3);
        
        if length(unique(s))==3
            % if s are different, use the first order derivative to
            % calculate the second order derivative
            c_bar = pdf_MF_normal(s,1);
            for i=1:3
                jk = setdiff(1:3,i);
                j = jk(1);
                k = jk(2);
                EQij = -(c_bar+dc_bar(i))*s(i)/(s(j)^2-s(i)^2) + ...
                    (c_bar+dc_bar(j))*s(j)/(s(j)^2-s(i)^2);
                EQik = -(c_bar+dc_bar(i))*s(i)/(s(k)^2-s(i)^2) + ...
                    (c_bar+dc_bar(k))*s(k)/(s(k)^2-s(i)^2);
                ddc_bar(i,i) = -2*dc_bar(i)-EQij-EQik;
                for j=i+1:3
                    k = setdiff(1:3,[i,j]);
                    EQij = -(c_bar+dc_bar(i))*s(j)/(s(j)^2-s(i)^2) + ...
                        (c_bar+dc_bar(j))*s(i)/(s(j)^2-s(i)^2);
                    EQkk = c_bar+dc_bar(k);
                    ddc_bar(i,j) = -c_bar-dc_bar(i)-dc_bar(j)+EQkk+EQij;
                    ddc_bar(j,i) = ddc_bar(i,j);
                end
            end
        else
            for i=1:3
                index=circshift([1 2 3],[0 4-i]);
                j=index(2);
                k=index(3);

                c_bar=pdf_MF_normal(s,1);
                ddc_bar(k,k) = integral(@(u) f_kunze_s_deriv_scaled_kk(u,[s(i),s(j),s(k)]),-1,1);
                ddc_bar(i,j) = integral(@(u) f_kunze_s_deriv_scaled_ij(u,[s(i),s(j),s(k)]),-1,1) ...
                    -dc_bar(i)-dc_bar(j)-c_bar;
                ddc_bar(j,i) = ddc_bar(i,j);
            end
        end
        
        varargout{2}=ddc_bar;
    end
end
end

function Y=f_kunze_s_deriv_scaled(u,s)
% integrand for the derivative of the scaled normalizing constant

J=besseli(0,1/2*(s(1)-s(2))*(1-u),1).*besseli(0,1/2*(s(1)+s(2))*(1+u),1);
Y=1/2*J.*(u-1).*exp((min([s(1) s(2)])+s(3))*(u-1));

end


function Y=f_kunze_s_deriv_scaled_kk(u,s)
% integrand for the second order derivative of the scaled normalizing constant

J=besseli(0,1/2*(s(1)-s(2))*(1-u),1).*besseli(0,1/2*(s(1)+s(2))*(1+u),1);
Y=1/2*J.*(u-1).^2.*exp((min([s(1) s(2)])+s(3))*(u-1));

end

function Y=f_kunze_s_deriv_scaled_ij(u,s)
% integrand for the scaled second order derivative of the normalizing constant

J=1/4*besseli(1,1/2*(s(2)-s(3))*(1-u),1).*besseli(0,1/2*(s(2)+s(3))*(1+u),1) ...
    .*exp((s(1)+min([s(2) s(3)]))*(u-1)).*u.*(1-u) ...
    +1/4*besseli(0,1/2*(s(2)-s(3))*(1-u),1).*besseli(1,1/2*(s(2)+s(3))*(1+u),1) ...
    .*exp((s(1)+min([s(2) s(3)]))*(u-1)).*u.*(1+u);
Y=J;

end

function Y=f_kunze_s_deriv_i(u,s,i)
% integrand for the derivative of the normalizing constant
index=circshift([1 2 3],[0 4-i]);
j=index(2);
k=index(3);

J00=besseli(0,1/2*(s(j)-s(k))*(1-u)).*besseli(0,1/2*(s(j)+s(k))*(1+u));
Y=1/2*J00.*u.*exp(s(i)*u);

end

function Y=f_kunze_s_deriv_ii(u,s,i)
% integrand for the second-order derivative of the normalizing constant
index=circshift([1 2 3],[0 4-i]);
j=index(2);
k=index(3);

J00=besseli(0,1/2*(s(j)-s(k))*(1-u)).*besseli(0,1/2*(s(j)+s(k))*(1+u));
Y=1/2*J00.*u.^2.*exp(s(i)*u);

end

function Y=f_kunze_s_deriv_ij(u,s,i,j)
% integrand for the mixed second-order derivative of the normalizing constant
k=setdiff([1 2 3],[i,j]);

J10=besseli(1,1/2*(s(j)-s(k))*(1-u)).*besseli(0,1/2*(s(j)+s(k))*(1+u));
J01=besseli(0,1/2*(s(j)-s(k))*(1-u)).*besseli(1,1/2*(s(j)+s(k))*(1+u));
Y=1/4*J10.*u.*(1-u).*exp(s(i)*u) + 1/4*J01.*u.*(1+u).*exp(s(i)*u);

end
