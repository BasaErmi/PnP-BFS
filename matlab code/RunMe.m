% Copyright by Lei Yang, The Hong Kong Polytechnic Univercity 
% The ADMM for Background/Foreground Extraction, January, 2013
% Solve the nonconvex problem, for 0<p<1 and \lambda > 0  
% min lambda*\|S\|_p^p + (1/2)*\|D - L - S\|_F^2
% s.t. L \in \mathcal{L},
% where \mathcal{L}={L | l_1 = l_2 = ... = l_n}

clear; % clc

ranseed = 1;
rand('seed', ranseed);
randn('seed', ranseed);

%% problem setup
load hall.mat
D = data;
mu = 1e-2;                 % penalty parameter
p = 0.6;

%% parameters settings 
opts.tau = 0.8;                  % the dual step size
opts.tol = 1e-4;                % the stop criterion
opts.InnerTol = 5e-3;           % the inner stop criterion of ADMM
opts.maxiter = 2000;             % maximal number of iterations
opts.identity = 1;               % the option of identity operator
opts.blurring = 0;               % the option of blurred operator
opts.heuristic = 1;              % the option of using heuristic method to choose beta
opts.display = 1;                % the option of displaying the results in the algorithm
opts.displayfreq = 1;            % the frequence of printing the results

% the option of blurred data
if opts.blurring == 1
    opts.lambda_max = lambda_max; opts.lambda_min = lambda_min;
    opts.Ar = Ar; opts.Ac = Ac; opts.picsize = picture_size;
end

%% call algorithm
tic
profile on
[L_out, S_out, Iter_out] = ADMM_lp(D, mu, p, opts);
% [L_out, S_out, Iter_out] = PALM_lp(D, mu, p, opts);
profile off
elapsed_time = toc;

%% process the result
if opts.identity == 1
    f_val = mu*norm(S_out(:), p)^p + (1/2)*norm(D(:)-L_out(:)-S_out(:))^2;
elseif opts.blurring == 1
    HLS = Get_HZ(L_out+S_out, Ac, Ar, picture_size);
    f_val = mu*norm(S(:), p)^p + (1/2)*norm(D(:)-HLS(:))^2;
end
S_mask = abs(S_out) > 1e-3;
spr = length(find(S_out~=0))/numel(D);
foreground = reshape(S_mask(:,label), picture_size);
ind_f = find(foreground == 1);
ind_g = find(groundtruth == 1);
ind_correct = intersect(ind_f, ind_g);
precision = length(ind_correct)/length(ind_f);
recall = length(ind_correct)/length(ind_g);
F = 2*precision*recall/(precision+recall);

%% display the result
fprintf('\n$p$ & $mu$ & $tau$ & Iter & Time(s) & spr & F-measure & f_val   \\\\ \n')
fprintf('  %g  &  %0.0e  & %0.1f & %i & %0.2f & %0.4f & %0.4f & %0.4f\n', p, mu, opts.tau, Iter_out, elapsed_time, spr, F, f_val);

% figure; imshow(reshape(D(:,label), picture_size), []);
% figure; imshow(reshape(L_out(:,label), picture_size), []);
% figure; imshow(reshape(S_out(:,label), picture_size), []);



