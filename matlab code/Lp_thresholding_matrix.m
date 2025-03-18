function [X, iter] = Lp_thresholding_matrix(Y, lambda, p)
% compute the matrix Lq regularized projection problem
% min f(X) := lambda*||X||_p^p + 1/2 * ||X - Y||_F^2 (0 < p <= 1)

iter = 0;
if p == 1
    X = sign(Y) .* max(abs(Y) - lambda, 0);
else
    B = abs(Y); X = zeros(size(B));
    tau = (2*lambda*(1 - p))^(1/(2-p)) + lambda*p*(2*lambda*(1 - p))^((p-1)/(2-p));
    % Using Newton method to find the minimizer in (xl, +\infty), where x1 = (2*p*(1 - p)*lambda)^(1/(2-p))
    index = find(B > tau);
    tmpX = B(index); tmpY = B(index);
    gX = tmpX - tmpY + lambda*p*tmpX.^(p-1);
    while max(abs(gX)) > 1e-12
        hX = 1 + lambda*p*(p-1)*tmpX.^(p-2);
        tmpX = tmpX - gX./hX;
        gX = tmpX - tmpY + lambda*p*tmpX.^(p-1);
        iter = iter + 1;
    end
    X(index) = tmpX.*sign(Y(index));
end

