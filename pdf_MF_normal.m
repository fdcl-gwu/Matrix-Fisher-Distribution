function c_return=pdf_MF_normal(s,bool_scaled)
%pdf_MF_normal: the normalizing constant for the matrix Fisher distribution
%on SO(3)
%   c = pdf_MF_normal(s) is the normalizing constant for the 
%   matrix Fisher distribution on SO(3), for a given 3x1 (or 1x3) proper singular
%   values s.
%
%   c = pdf_MF_normal(s,BOOL_SCALED) returns an exponentially scaled value 
%   specified by BOOL_SCALED:
%       0 - (defalut) is the same as pdf_MF_normal(s)
%       1 - returnes an exponentially scaled normlaizing constant,
%       exp(-sum(s))*c
%
%   See T. Lee, "Bayesian Attitude Estimation with the Matrix Fisher
%   Distribution on SO(3)", 2017, http://arxiv.org/abs/1710.03746
%
%   See also PDF_MF_NORMAL_APPROX

assert(or(min(size(s)==[1 3]),min(size(s)==[3 1])),'ERROR: s should be 3 by 1 or 1 by 3');

% if bool_scaled is not defined, then set it false
if nargin < 2
    bool_scaled=false;
end

if ~bool_scaled
    % return the normalizing constant without any scaling
    c = integral(@(u) f_kunze_s(u,s),-1,1);
    c_return = c;
else
    % return the normalizing constant scaled by exp(-sum(s))
    if s(1)>= s(2)
        c_bar = integral(@(u) f_kunze_s_scaled_1(u,s),-1,1);
    else
        c_bar = integral(@(u) f_kunze_s_scaled_2(u,s),-1,1);
    end
    c_return = c_bar;
    %c=c_bar*exp(sum(s));    
end

end

function Y=f_kunze_s(u,s)
% integrand for the normalizing constant

J=besseli(0,1/2*(s(1)-s(2))*(1-u)).*besseli(0,1/2*(s(1)+s(2))*(1+u));
Y=1/2*exp(s(3)*u).*J;

end

function Y=f_kunze_s_scaled_1(u,s)
% integrand for the normalizing constant scaled by exp(-sum(s)) when s(1)
% >= s(2)

J=besseli(0,1/2*(s(1)-s(2))*(1-u),1).*besseli(0,1/2*(s(1)+s(2))*(1+u),1);
Y=1/2*exp((s(2)+s(3))*(u-1)).*J;

end

function Y=f_kunze_s_scaled_2(u,s)
% integrand for the normalizing constant scaled by exp(-sum(s)) when s(1)
% <= s(2)

J=besseli(0,1/2*(s(1)-s(2))*(1-u),1).*besseli(0,1/2*(s(1)+s(2))*(1+u),1);
Y=1/2*exp((s(1)+s(3))*(u-1)).*J;

end
