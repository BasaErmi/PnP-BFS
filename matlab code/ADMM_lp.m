function [L_sol, S_sol, Iter_sol] = ADMM_lp(D, mu, p, opts)
% Call three blocks ADMM to solve the nonconvex problem,
% min mu*\|S\|_p^p + (1/2)*\|D - \scr{A}(Z)\|_F^2
% s.t. L \in \mathcal{C}, Z = L + S,
% where \mathcal{C}={L | l_1 = l_2 = ... = l_n, \|L\|_{\infty} \leq l}

%% parameter settings
if isfield(opts, 'tau'),            tau = opts.tau;                    else     tau = 0.8;             end    % the dual step size
if isfield(opts, 'upbd'),           upbd = opts.upbd;                  else     upbd = 1;              end    % the upper bound of the energy of each pixel
if isfield(opts, 'tol'),            OuterTol = opts.tol;               else     OuterTol = 1e-4;       end    % the outer stopping criterion
if isfield(opts, 'InnerTol'),       InnerTol = opts.InnerTol;          else     InnerTol = 5e-3;       end    % the inner stopping criterion
if isfield(opts, 'maxiter'),        maxiter = opts.maxiter;            else     maxiter = 1000;        end    % the maximum iteration number
if isfield(opts, 'identity'),       identity = opts.identity;          else     identity = 1;          end    % the option of identity operator
if isfield(opts, 'blurring'),       blurring = opts.blurring;          else     blurring = 0;          end    % the option of blurring operator
if isfield(opts, 'heuristic'),      heuristic = opts.heuristic;        else     heuristic = 1;         end    % the option of using heuristic method to choose beta
if isfield(opts, 'display'),        display = opts.display;            else     display = 1;           end    % the option of displaying the results in the algorithm
if isfield(opts, 'displayfreq'),    displayfreq = opts.displayfreq;    else     displayfreq = 1;       end    % the gap of display

if identity == 1
    lambda_max = 1; lambda_min = 1;
elseif blurring == 1
    if ~isfield(opts, 'lambda_max'),    error('Please input lambda_max');      end
    if ~isfield(opts, 'lambda_min'),    error('Please input lambda_min');      end
    if ~isfield(opts, 'Ar'),            error('Please input Ar');              end
    if ~isfield(opts, 'Ac'),            error('Please input Ac');              end
    if ~isfield(opts, 'picsize'),       error('Please input picsize');         end
    lambda_max = opts.lambda_max; lambda_min = opts.lambda_min;
    Ar = opts.Ar; Ac = opts.Ac; picsize = opts.picsize;
    [~, Sc, Vc] = svd(Ac); [~, Sr, Vr] = svd(Ar);
end

% the low bound of beta
beta_bound = max([lambda_max/tau, lambda_max*tau,...
    -0.5*lambda_min+0.5*sqrt(lambda_min^2 + 8*lambda_max^2/tau),...
    -0.5*lambda_min+0.5*sqrt(lambda_min^2 + 8*tau^2*lambda_max^2/(1+tau-tau^2))]);

% the initial beta
if heuristic == 1
    if isfield(opts, 'sigma1'),   sigma1 = opts.sigma1;     else sigma1 = 0.6;   end    % the parameter used to determinate initial beta0: opt.beta0 = opt.sigma1*opt.beta;
    if isfield(opts, 'sigma2'),   sigma2 = opts.sigma2;     else sigma2 = 0.3;   end    % the heuristic factor: count >= ceil(opt.sigma2*iter)
    if isfield(opts, 'rho_beta'), rho_beta = opts.rho_beta; else rho_beta = 1.1; end    % the increase fact of beta in heuristic method
    beta = sigma1*beta_bound; beta_max = 1.01*beta_bound; count = 0;
else
    beta = 1.01*beta_bound;
end

% initialize algorithm
[m, n] = size(D);
if isfield(opts, 'L0') && isfield(opts, 'S0') && isfield(opts, 'Z0') && isfield(opts, 'Lambda0')
    L = opts.L0; S = opts.S0; Z = opts.Z0; Lambda = opts.Lambda0;
else
    kappa = 1; 
    PcD = repmat(mean(kappa*D, 2), 1, n); 
    PcD = sign(PcD).*min(abs(PcD), upbd);
    L = PcD; S = zeros(m, n); Z = L;
    if identity == 1
        Lambda = D - Z;
    elseif blurring == 1;
        Lambda = [];
        for il = 1 : n
            Lam_tmp = Ac'*(reshape(D(:,il),picsize) - Ac*reshape(Z(:,il),picsize)*Ar')*Ar;
            Lambda = [Lambda Lam_tmp(:)];
        end
    end
end

if display == 1
    fprintf('\n------------ 3-blocks ADMM for nonconvex background model ---------------\n');
    fprintf('p = %0.1f,   tau = %0.1f \n', p, tau);
    fprintf('iter   |   relerr   |   iter_Newton   |   f_val   |   Phi_value   |  Gap_Phi  |  beta\n');
end

succ_chg = inf;

%% run algorithm
for iter = 1 : maxiter
    
    Lk = L; Sk = S; Zk = Z; Lambdak = Lambda; succ_chgk = succ_chg;
    
    %% L update step
    Tmp1 = Z + Lambda/beta;
    L_tmp = Tmp1 - S;
    L = repmat(mean(L_tmp, 2), 1, n);
    L = sign(L).*min(abs(L), upbd);
    
    %% S update step
    S_tmp = Tmp1 - L;
    [S, iter_Newton] = Lp_thresholding_matrix(S_tmp, mu/beta, p);
    
    %% Z update step
    if identity == 1 
        Z = ( D + beta*(L + S) - Lambda ) / (1 + beta);   
    elseif blurring == 1
        Z_tmp1 = beta*(L + S) - Lambda;
        Sig = beta + diag(Sc).^2 * diag(Sr).^2';
        for iz = 1 : n
            Z_tmp2 = Ac'*reshape(D(:,iz),picsize)*Ar + reshape(Z_tmp1(:,iz),picsize);
            Z_tmp3 = Vc*((Vc'*Z_tmp2*Vr)./Sig)*Vr';
            Z(:, iz) = Z_tmp3(:);
            % lhs = beta*Z_tmp3 + Ac'*Ac*Z_tmp3*Ar'*Ar;
            % fprintf('left - right = %g\n', norm(lhs-Z_tmp2,'fro'));
        end      
    end
    
    %% update multiplier
    Lambda = Lambda - tau * beta * (L + S - Z);
    
    %% compute relative error
    succ_chg = norm(L-Lk,'fro') + norm(Z-Zk,'fro');
    fnorm = norm(L,'fro') + norm(Z,'fro') + 1;
    relchg = succ_chg/fnorm;
    
    if relchg < OuterTol
        
        succ_chg_2nd = norm(S-Sk,'fro') + norm(Lambda-Lambdak,'fro');
        fnorm_2nd = norm(S,'fro') + norm(Lambda,'fro') + 1;
        relchg_2nd = succ_chg_2nd/fnorm_2nd;
        
        if relchg_2nd < InnerTol
            if display == 1
                if identity == 1
                    Phi_valk = mu*norm(Sk(:), p)^p + (1/2)*norm(D(:)-Zk(:))^2 - (Lambdak(:))'*(Lk(:)+Sk(:)-Zk(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(Lk(:)+Sk(:)-Zk(:))^2;
                    Phi_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-Z(:))^2 - (Lambda(:))'*(L(:)+S(:)-Z(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(L(:)+S(:)-Z(:))^2;
                    Gap_Phi = Phi_valk - Phi_val;
                    f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-L(:)-S(:))^2;
                elseif blurring == 1
                    HZk = Get_HZ(Zk, Ac, Ar, picsize); HZ = Get_HZ(Z, Ac, Ar, picsize);
                    Phi_valk = mu*norm(Sk(:), p)^p + (1/2)*norm(D(:)-HZk(:))^2 - (Lambdak(:))'*(Lk(:)+Sk(:)-Zk(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(Lk(:)+Sk(:)-Zk(:))^2;
                    Phi_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HZ(:))^2 - (Lambda(:))'*(L(:)+S(:)-Z(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(L(:)+S(:)-Z(:))^2;
                    Gap_Phi = Phi_valk - Phi_val;
                    HLS = Get_HZ(L+S, Ac, Ar, picsize);
                    f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HLS(:))^2;
                end
                fprintf(' %i | %0.8e | %i | %0.8e | %0.8e | %0.8e | %g\n', iter, relchg, iter_Newton, f_val, Phi_val, Gap_Phi, beta);
            end
            break
        end
    end
    
    % Print the results
    if display == 1
        if mod(iter, displayfreq) == 0
            if identity == 1
                Phi_valk = mu*norm(Sk(:), p)^p + (1/2)*norm(D(:)-Zk(:))^2 - (Lambdak(:))'*(Lk(:)+Sk(:)-Zk(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(Lk(:)+Sk(:)-Zk(:))^2;
                Phi_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-Z(:))^2 - (Lambda(:))'*(L(:)+S(:)-Z(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(L(:)+S(:)-Z(:))^2;
                Gap_Phi = Phi_valk - Phi_val;
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-L(:)-S(:))^2; 
            elseif blurring == 1
                HZk = Get_HZ(Zk, Ac, Ar, picsize); HZ = Get_HZ(Z, Ac, Ar, picsize);
                Phi_valk = mu*norm(Sk(:), p)^p + (1/2)*norm(D(:)-HZk(:))^2 - (Lambdak(:))'*(Lk(:)+Sk(:)-Zk(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(Lk(:)+Sk(:)-Zk(:))^2;
                Phi_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HZ(:))^2 - (Lambda(:))'*(L(:)+S(:)-Z(:)) + (beta/2+max(1-tau, (tau-1)*tau^2/(1+tau-tau^2)))*norm(L(:)+S(:)-Z(:))^2;
                Gap_Phi = Phi_valk - Phi_val;
                HLS = Get_HZ(L+S, Ac, Ar, picsize);
                f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HLS(:))^2;
            end
            fprintf(' %i | %0.8e | %i | %0.8e | %0.8e | %0.8e | %g\n', iter, relchg, iter_Newton, f_val, Phi_val, Gap_Phi, beta);
        end
    end
    
    %% heuristic of changing the value of beta
    if heuristic == 1
        if (0.99*succ_chgk - succ_chg) < 0
            count = count + 1;
        end
        if count >= ceil(sigma2*iter) || fnorm > 1e10
            beta = min(beta*rho_beta, beta_max);
        end
    end
    
    
end

L_sol = L;
S_sol = S;
Iter_sol = iter;





