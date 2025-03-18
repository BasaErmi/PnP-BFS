function [L_sol, S_sol, Iter_sol] = PALM_lp(D, mu, p, opts)
% Call Proximal Alternating Linearized Minimization to solve the nonconvex problem,
% min mu*\|S\|_p^p + (1/2)*\|D - \scr{A}(L + S)\|_F^2
% s.t. L \in \mathcal{C},
% where \mathcal{C}={L | l_1 = l_2 = ... = l_n, \|L\|_{\infty} \leq l}


%% parameter settings
if isfield(opts, 'upbd'),           upbd = opts.upbd;                  else     upbd = 1;              end    % the upper bound of the energy of each pixel
if isfield(opts, 'tol'),            Tol = opts.tol;                    else     Tol = 1e-4;            end    % the stopping criterion
if isfield(opts, 'maxiter'),        maxiter = opts.maxiter;            else     maxiter = 1000;        end    % the maximum iteration number
if isfield(opts, 'identity'),       identity = opts.identity;          else     identity = 1;          end    % the option of identity operator
if isfield(opts, 'blurring'),       blurring = opts.blurring;          else     blurring = 0;          end    % the option of blurring operator
if isfield(opts, 'display'),        display = opts.display;            else     display = 1;           end    % the option of displaying the results in the algorithm
if isfield(opts, 'displayfreq'),    displayfreq = opts.displayfreq;    else     displayfreq = 1;       end    % the gap of display

if identity == 1
    lambda_max = 1;
elseif blurring == 1
    if ~isfield(opts, 'lambda_max'), error('Please input lambda_max');  end
    if ~isfield(opts, 'lambda_min'), error('Please input lambda_min');  end
    if ~isfield(opts, 'Ar'),         error('Please input Ar');          end
    if ~isfield(opts, 'Ac'),         error('Please input Ac');          end
    if ~isfield(opts, 'picsize'),    error('Please input picsize');     end
    Ar = opts.Ar; Ac = opts.Ac; lambda_max = opts.lambda_max; picsize = opts.picsize;
end

[m, n] = size(D);
c = lambda_max/0.99;
d = lambda_max/0.99;

% initialize algorithm
if isfield(opts, 'L0') && isfield(opts, 'S0')
    L = opts.L0; S = opts.S0;
else
    L = zeros(m, n); S = zeros(m, n);
    % L = repmat(mean(D, 2),1,n(2)); S = zeros(n);
end

if display == 1
    fprintf('\n------------ PALM for nonconvex background model with p = %0.1f ---------------------\n\n', p);
    fprintf('iter   |   relerr   |   iter_Newton   |   f_val\n');
end

%% run algorithm
for iter = 1 : maxiter
    
    Lk = L; Sk = S;
    
    if identity == 1
        
        % L update step
        L_tmp = L - (1/c)*((L + S) - D);
        L = repmat(mean(L_tmp, 2), 1, n);
        L = sign(L).*min(abs(L), upbd);
        % S update step
        S_tmp = S - (1/d)*((L + S) - D);
        [S, iter_Newton] = Lp_thresholding_matrix(S_tmp, mu/d, p);
        
    elseif blurring == 1
        
        % L update step
        L1 = [];
        for il = 1 : n
            l_tmp = Ac'*(Ac*reshape(L(:,il)+S(:,il),picsize)*Ar' - reshape(D(:,il),picsize))*Ar;
            L1 = [L1 l_tmp(:)];
        end
        L2 = L - (1/c)*L1; 
        L = repmat(mean(L2, 2), 1, n);
        L = sign(L).*min(abs(L), upbd);
        
        % S update step
        S1 = [];
        for is = 1 : n
            s_tmp = Ac'*(Ac*reshape(L(:,is)+S(:,is),picsize)*Ar' - reshape(D(:,is),picsize))*Ar;
            S1 = [S1 s_tmp(:)];
        end
        S2 = S - (1/d)*S1;
        [S, iter_Newton] = Lp_thresholding_matrix(S2, mu/d, p);
        
    end
    
    %% compute relative error
    succ_chg = norm(L-Lk,'fro') + norm(S-Sk,'fro');
    fnorm = norm(L,'fro') + norm(S,'fro') + 1;
    relchg = succ_chg/fnorm;
    
    if relchg < Tol
        if display == 1
            if identity == 1
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-L(:)-S(:))^2;
            elseif blurring == 1
                HLS = Get_HZ(L+S, Ac, Ar, picsize);
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HLS(:))^2;
            end
            fprintf('  %i   |  %0.8e  |  %i  |   %0.8e\n', iter, relchg, iter_Newton, f_val);
        end
        break
    end
    
    % Print the results
    if display == 1
        if mod(iter, displayfreq) == 0
            if identity == 1
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-L(:)-S(:))^2;
            elseif blurring == 1
                HLS = Get_HZ(L+S, Ac, Ar, picsize);
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HLS(:))^2;
            end
            fprintf('  %i   |  %0.8e  |  %i  |   %0.8e\n', iter, relchg, iter_Newton, f_val);
        end
    end
    
    
end

L_sol = L;
S_sol = S;
Iter_sol = iter;






