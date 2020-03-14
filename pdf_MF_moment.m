function varargout=pdf_MF_moment(s,bool_M2,bool_M3)
%pdf_MF_moment: the canonical moment of the matrix Fisher distribution on SO(3)
%   [M1, M2, c_bar] = pdf_MF_normal(s,BOOL_M2) returns the 3x1 matrix M1 of 
%   the first moments and the 3x3 matrix of the non-zero second order moments 
%   for the matrix Fisher distribution with the parameter S=diag(s). 
%   It also returns the exponentially scaled normalizing constant. 
%
%   To obtain the 3x3 first moment for an arbitrary matrix parameter F, run 
%       [U S V]=psvd(F);
%       M1=U*pdf_MF_normal(diag(S))*V';
%
%   BOOL_M2 determines whether the second order canonical moments
%   are computed or not:
%       0 - (defalut) is the same as [M1, c_bar]=pdf_MF_moment(s), and the 
%       second order moments are not computed
%       1 - computes the second order moments
%
%   See T. Lee, "Bayesian Attitude Estimation with the Matrix Fisher
%   Distribution on SO(3)", 2017, http://arxiv.org/abs/1710.03746


if nargin < 2
    bool_M2 = false;
end
if nargin < 3
    bool_M3 = false;
end

% compute the first order moments
c_bar=pdf_MF_normal(s,1);
dc_bar=pdf_MF_normal_deriv(s,0,1);
M1=dc_bar/c_bar+1;

if ~bool_M2
    varargout{1}=M1;
    varargout{2}=c_bar;
else
    % compute the second order moments
    M2 = zeros(3,3);
    for i = 1:3
        for j = setdiff(1:3,i)
            M2(3*(i-1)+j,3*(i-1)+j) = M1(i)*s(i)/(s(i)^2-s(j)^2)-M1(j)*s(j)/(s(i)^2-s(j)^2);
            M2(3*(i-1)+j,3*(j-1)+i) = M1(i)*s(j)/(s(i)^2-s(j)^2)-M1(j)*s(i)/(s(i)^2-s(j)^2);
        end
    end

    M2(1,1) = 1-M2(2,2)-M2(3,3);
    M2(5,5) = 1-M2(4,4)-M2(6,6);
    M2(9,9) = 1-M2(7,7)-M2(8,8);

    M2(1,5) = M2(2,4)+M1(3);
    M2(1,9) = M2(3,7)+M1(2);
    M2(5,9) = M2(6,8)+M1(1);
    M2(5,1) = M2(1,5);
    M2(9,1) = M2(1,9);
    M2(9,5) = M2(5,9);
    
    if ~bool_M3
        varargout{1}=M1;
        varargout{2}=M2;
        varargout{3}=c_bar;
    else
        % compute the third order moments
        M3 = zeros(9,9,9);
        for i = 1:3
            for j = 1:3
                for k = setdiff(1:3,j)
                    if i~=j && i~=k
                        M3(3*(i-1)+i,3*(j-1)+k,3*(j-1)+k) = M2(3*(i-1)+i,3*(j-1)+j)*s(j)/(s(j)^2-s(k)^2)...
                            - M2(3*(i-1)+i,3*(k-1)+k)*s(k)/(s(j)^2-s(k)^2);
                        M3(3*(i-1)+i,3*(j-1)+k,3*(k-1)+j) = M2(3*(i-1)+i,3*(j-1)+j)*s(k)/(s(j)^2-s(k)^2)...
                            - M2(3*(i-1)+i,3*(k-1)+k)*s(j)/(s(j)^2-s(k)^2);

                        M3(3*(j-1)+k,3*(i-1)+i,3*(j-1)+k) = M3(3*(i-1)+i,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(i-1)+i,3*(k-1)+j) = M3(3*(i-1)+i,3*(j-1)+k,3*(k-1)+j);

                        M3(3*(j-1)+k,3*(j-1)+k,3*(i-1)+i) = M3(3*(i-1)+i,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(k-1)+j,3*(i-1)+i) = M3(3*(i-1)+i,3*(j-1)+k,3*(k-1)+j);
                    end

                    if i==j
                        M3(3*(j-1)+j,3*(j-1)+k,3*(j-1)+k) = M2(3*(j-1)+j,3*(j-1)+j)*s(j)/(s(j)^2-s(k)^2) ...
                            - M2(3*(j-1)+j,3*(k-1)+k)*s(k)/(s(j)^2-s(k)^2) ...
                            - M1(j)*(s(j)^2+s(k)^2)/(s(j)^2-s(k)^2)^2 ...
                            + M1(k)*2*s(j)*s(k)/(s(j)^2-s(k)^2)^2;
                        M3(3*(j-1)+j,3*(j-1)+k,3*(k-1)+j) = M2(3*(j-1)+j,3*(j-1)+j)*s(k)/(s(j)^2-s(k)^2) ...
                            - M2(3*(j-1)+j,3*(k-1)+k)*s(j)/(s(j)^2-s(k)^2) ...
                            - M1(j)*2*s(j)*s(k)/(s(j)^2-s(k)^2)^2 ...
                            + M1(k)*(s(j)^2+s(k)^2)/(s(j)^2-s(k)^2)^2;

                        M3(3*(j-1)+k,3*(j-1)+j,3*(j-1)+k) = M3(3*(j-1)+j,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(j-1)+j,3*(k-1)+j) = M3(3*(j-1)+j,3*(j-1)+k,3*(k-1)+j);

                        M3(3*(j-1)+k,3*(j-1)+k,3*(j-1)+j) = M3(3*(j-1)+j,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(k-1)+j,3*(j-1)+j) = M3(3*(j-1)+j,3*(j-1)+k,3*(k-1)+j);
                    end

                    if i==k
                        M3(3*(k-1)+k,3*(j-1)+k,3*(j-1)+k) = M2(3*(k-1)+k,3*(k-1)+k)*s(k)/(s(k)^2-s(j)^2) ...
                            - M2(3*(k-1)+k,3*(j-1)+j)*s(j)/(s(k)^2-s(j)^2) ...
                            - M1(k)*(s(k)^2+s(j)^2)/(s(k)^2-s(j)^2)^2 ...
                            + M1(j)*2*s(k)*s(j)/(s(k)^2-s(j)^2)^2;
                        M3(3*(k-1)+k,3*(j-1)+k,3*(k-1)+j) = M2(3*(k-1)+k,3*(k-1)+k)*s(j)/(s(k)^2-s(j)^2) ...
                            - M2(3*(k-1)+k,3*(j-1)+j)*s(k)/(s(k)^2-s(j)^2) ...
                            - M1(k)*2*s(k)*s(j)/(s(k)^2-s(j)^2)^2 ...
                            + M1(j)*(s(k)^2+s(j)^2)/(s(k)^2-s(j)^2)^2;

                        M3(3*(j-1)+k,3*(k-1)+k,3*(j-1)+k) = M3(3*(k-1)+k,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(k-1)+k,3*(k-1)+j) = M3(3*(k-1)+k,3*(j-1)+k,3*(k-1)+j);

                        M3(3*(j-1)+k,3*(j-1)+k,3*(k-1)+k) = M3(3*(k-1)+k,3*(j-1)+k,3*(j-1)+k);
                        M3(3*(j-1)+k,3*(k-1)+j,3*(k-1)+k) = M3(3*(k-1)+k,3*(j-1)+k,3*(k-1)+j);
                    end
                end
            end
        end

        for i = 1:3
            for j = setdiff([1,2,3],i)
                for k = setdiff([1,2,3],[i,j])
                    M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i) = M1(i)*s(j)*s(k)/(s(i)^2-s(j)^2)/(s(i)^2-s(k)^2)...
                        + M1(j)*s(i)*s(k)/(s(j)^2-s(i)^2)/(s(j)^2-s(k)^2)...
                        + M1(k)*s(i)*s(j)/(s(k)^2-s(i)^2)/(s(k)^2-s(j)^2);
                    M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k) = M1(i)*s(j)*s(i)/(s(i)^2-s(j)^2)/(s(i)^2-s(k)^2)...
                        + M1(j)*s(j)^2/(s(j)^2-s(i)^2)/(s(j)^2-s(k)^2)...
                        + M1(k)*s(k)*s(j)/(s(k)^2-s(i)^2)/(s(k)^2-s(j)^2);

                    % ijkijk & ijikjk
                    M3(3*(i-1)+j,3*(k-1)+i,3*(j-1)+k) = M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i);
                    M3(3*(i-1)+j,3*(i-1)+k,3*(j-1)+k) = M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k);

                    % jkijki & jkijik
                    M3(3*(j-1)+k,3*(i-1)+j,3*(k-1)+i) = M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i);
                    M3(3*(j-1)+k,3*(i-1)+j,3*(i-1)+k) = M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k);

                    % jkkiij & jkikij
                    M3(3*(j-1)+k,3*(k-1)+i,3*(i-1)+j) = M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i);
                    M3(3*(j-1)+k,3*(i-1)+k,3*(i-1)+j) = M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k);

                    % kiijjk & ikijjk
                    M3(3*(k-1)+i,3*(i-1)+j,3*(j-1)+k) = M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i);
                    M3(3*(i-1)+k,3*(i-1)+j,3*(j-1)+k) = M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k);

                    % kijkij & ikjkij
                    M3(3*(k-1)+i,3*(j-1)+k,3*(i-1)+j) = M3(3*(i-1)+j,3*(j-1)+k,3*(k-1)+i);
                    M3(3*(i-1)+k,3*(j-1)+k,3*(i-1)+j) = M3(3*(i-1)+j,3*(j-1)+k,3*(i-1)+k);
                end
            end
        end

        M3(1,1,1) = M1(1)-M3(1,2,2)-M3(1,3,3);
        M3(1,1,5) = M3(1,2,4)+M2(1,9);
        M3(1,1,9) = M3(1,3,7)+M2(1,5);
        M3(1,5,5) = M1(1)-M3(1,4,4)-M3(1,6,6);
        M3(1,5,9) = M3(1,6,8)+M2(1,1);
        M3(1,9,9) = M1(1)-M3(1,7,7)-M3(1,8,8);
        M3(5,5,5) = M1(2)-M3(5,4,4)-M3(5,6,6);
        M3(5,5,9) = M3(5,6,8)+M2(5,1);
        M3(5,9,9) = M1(2)-M3(5,7,7)-M3(5,8,8);
        M3(9,9,9) = M1(3)-M3(9,7,7)-M3(9,8,8);

        M3(1,5,1) = M3(1,1,5);
        M3(5,1,1) = M3(1,1,5);

        M3(1,9,1) = M3(1,1,9);
        M3(9,1,1) = M3(1,1,9);

        M3(5,1,5) = M3(1,5,5);
        M3(5,5,1) = M3(1,5,5);

        M3(1,9,5) = M3(1,5,9);
        M3(5,1,9) = M3(1,5,9);
        M3(5,9,1) = M3(1,5,9);
        M3(9,1,5) = M3(1,5,9);
        M3(9,5,1) = M3(1,5,9);

        M3(9,1,9) = M3(1,9,9);
        M3(9,9,1) = M3(1,9,9);

        M3(5,9,5) = M3(5,5,9);
        M3(9,5,5) = M3(5,5,9);

        M3(9,5,9) = M3(5,9,9);
        M3(9,9,5) = M3(5,9,9);
        
        varargout{1}=M1;
        varargout{2}=M2;
        varargout{3}=M3;
        varargout{4}=c_bar;
    end
end

end
