% =========================================================================
% Tensor Factorization: Learning Mixtures of Gaussians
% ------------------
% Paper: Tensor Decompositions and their Applications in Machine Learning
% Seminar: Efficient Inference and Large Scale Machine Learning
% Author: Stephan Rabanser (rabanser@cs.tum.edu)
% =========================================================================

% -----------------------
% Configuration and setup
% -----------------------

close all; clear; clc;

d = 3;              % # of dimensions
k = 2;              % # of clusters
n = 10000;          % # of samples per cluster
tot = k * n;        % # of total samples
s = 2;              % isotropic variance
dist = 40;          % distance between gaussians
spher = 1;          % decide shape of the covariance matrix
spher_range = 2;    % range of the values of the covariance matrix
csv_import = 0;     % decide whether data should be imported

assert(k <= d, 'k cannot be larger than d')

rand('seed',11);    % initilize random generator for reproducibility 

% --------------------
% Generate sample data
% --------------------

if csv_import == true
    fprintf('Reading sample data ...\n');
    tic
    
    A = csvread('A.csv');
    X = csvread('X.csv');
    
    for i = 1:k
        mean = A(:,i)';
        scatter(X((i-1)*n+1:i*n,1),X((i-1)*n+1:i*n,2),10,'.')
        hold on
        scatter(mean(1),mean(2),100,'x', 'black')
        hold on
    end
else
    fprintf('Generating sample data ...\n');
    tic
    
    A = -dist+(dist+dist)*rand(d, k);           % mixture component means
    X = [];                                     % GMM data

    figure
    for i = 1:k
        mean = A(:,i)';
        if spher == true
            covariance = s * eye(d);
        else 
            a = -spher_range+(spher_range+spher_range)*rand(d,d);
            covariance = a'* a;
        end
        mvn = mvnrnd(mean, covariance, n);
        scatter(mvn(:,1),mvn(:,2),10,'.')
        hold on
        scatter(mean(1),mean(2),100,'x', 'black')
        hold on
        X = [X ; mvn];
    end
end

toc
fprintf('Data present.\n');

% ---------------------------
% Computate first two moments
% ---------------------------

fprintf('Compute data mean ...\n');
tic

mu = zeros(d,1);
for i = 1:tot
    mu = mu + X(i, :)';
end
mu = mu / tot;

toc
fprintf('Mean computed.\n');

fprintf('Compute data covariance ...\n');
tic

Sigma = zeros(d,d);
for i = 1:tot
    Sigma = Sigma + X(i, :)' * X(i, :);
end
Sigma = Sigma / tot;

toc
fprintf('Covariance computed.\n');

% ---------------------------------------
% Compute SVD from data covariance matrix
% ---------------------------------------

fprintf('Performing whitening ...\n');
tic

[U,S,~] = svd(Sigma);
s_est = S(end,end);                                             % Estimate sigma^2
fprintf('Estimating variance %d as %d. \n', s, s_est);
W = U(:,1:k) * sqrt(pinv(S(1:k,1:k)-diag(ones(k,1).*s_est)));   % Obtain whitening matrix
X_whit = X * W;                                                 % Whiten the data

toc
fprintf('Whitening performed.\n');

% ------------------------------------
% Compute whitenend third order moment
% ------------------------------------

% T = zeros(k,k,k);
% for t = 1 : tot
%     for i = 1 : k
%         for j = 1 : k
%             for l = 1 : k
%                 T(i,j,l) = T(i,j,l) + X_whit(t,i)*X_whit(t,j)*X_whit(t,l);
%             end
%         end
%     end
% end
% T = T / tot;

% ------------------------------------------------------------------------
% Apply tensor power method
% ------------------------------------------------------------------------
% Inspired by: https://bitbucket.org/kazizzad/tensor-power-method/overview
% ------------------------------------------------------------------------

fprintf('Performing tensor power method ...\n');
tic

TOL = 1e-8;                 % Convergence delta
maxiter = 100;              % Maximum number of power step iterations
V_est = zeros(k,k);         % Estimated eigenvectors for tensor V
lambda = zeros(k,1);        % Estimated eigenvalues for tensor V

for i = 1:k
    % Generate initial random vector
    v_old = rand(k,1);
    v_old = v_old./norm(v_old);
    for iter = 1 : maxiter
        % Perform multilinear transformation
        v_new = (X_whit'* ((X_whit* v_old) .* (X_whit* v_old)))/tot;
        v_new = v_new - s_est * (W' * mu * dot((W*v_old),(W*v_old)));
        v_new = v_new - s_est * (2 * W' * W * v_old * ((W'*mu)' * (v_old)));
        % Defaltion
        if i > 1 
            for j = 1:i-1
                v_new = v_new - (V_est(:,j) * (v_old'*V_est(:,j))^2)* lambda(j);
            end          
        end
        % Compute new eigenvalue and eigenvector
        l = norm(v_new);
        v_new = v_new./norm(v_new);
        % Check for convergence
        if norm(v_old - v_new) < TOL
            fprintf('Eigenvector %d converged at iteration %d. \n', i, iter);
            V_est(:,i) = v_new;
            lambda(i,1) = l;
            break;
        end
        v_old = v_new;
    end
end

toc
fprintf('Tensor power method performed.\n');

% ------------------------------
% Apply backwards transformation
% ------------------------------

fprintf('Perform backwards transformation ...\n');
tic

A_est = pinv(W') * V_est * diag(lambda);

toc
fprintf('Backwards transformation performed.\n');

fprintf('Plotting results ...\n');

for i = 1:k
    mean_est = A_est(:,i)';
    % Plot mean
    scatter(mean_est(1),mean_est(2),1000,'+', 'red')
    hold on
    % Plot variance
    viscircles([mean_est(1) mean_est(2)], sqrt(s_est));
    hold on
end
