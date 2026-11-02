%% simulation code
tic;

clearvars
close all
rng(1);

%% set parameters
N = 8; M = 16; NM = N * M;
PRF = 1500; lambda = 299792458/1.33e9;
vp = 75.5; d = lambda/2; thetaL = 0;

Ntheta = 360;
a = 1; b = 1;
CNR = 100;
Ktype = 1; Dtype = 0; u = 0;

p.radar  = struct('N', N, 'M', M, 'PRF', PRF, 'lambda', lambda, 'vp', vp, ...
    'd', d, 'thetaL', thetaL);
p.cModel = struct('Ntheta', Ntheta, 'a', a, 'b', b, 'CNR', CNR, 'Ktype', ...
    Ktype, 'Dtype', Dtype, 'u', u);
Nr = 1024;

%% verify brennan's rule
varsigma = gen_clutter_data(p, Nr);

beta = (2 * vp)/(PRF * d);
r_brennan = N + beta*(M - 1);

R_hat  = (varsigma * varsigma')/Nr;               % sample covariance
e      = sort(real(eig(R_hat)), 'descend');       % NM eigenvalues of R_hat in descending order
Pn_hat = mean(e(ceil(r_brennan)+5:end));          % noise floor from the tail (+5 is safety margin)
edge   = Pn_hat*(1 + sqrt(NM/Nr))^2;              % largest pure-noise eigenvalue expected
r_hat  = sum(e > edge);

figure; plot(10*log10(e/Pn_hat), 'o'); hold on
xline(r_brennan, '--', 'Brennan'); yline(10*log10(edge/Pn_hat), ':')
xlabel('Eigenvalue Index'); ylabel('dB Above Noise')

%% K-dist clutter simulation (shared texture (tau) across clutter patches of each range gate)
p.cModel.Ktype = 3;
p.cModel.a = 0.5; 
p.cModel.b = 1/p.cModel.a;

[varsigma, tau, c, G] = gen_clutter_data(p, Nr);

% a = 0.5
figure
subplot(1, 2, 1)
histogram(abs(varsigma(:)), 200, 'Normalization', 'pdf')

% overlay theoretical k-distribution curve
P = mean(abs(varsigma(:)).^2);
nu = p.cModel.a;
r = linspace(min(abs(varsigma(:))), max(abs(varsigma(:))), 200);

pdf_K = (4 * nu^((nu + 1)/2) * r.^(nu))/(gamma(nu) * P^((nu + 1)/2)) .* besselk(nu - 1, 2 * r * sqrt(nu/P));

hold on;
plot(r, pdf_K)

xlabel('Amplitude $r = |\varsigma|$', 'Interpreter', 'latex')
ylabel('$p(r)$', 'Interpreter', 'latex')
title(sprintf('$\\nu = %.1f$', p.cModel.a), 'Interpreter', 'latex')
legend('Simulated', 'K-distribution (theory)')

% a = 20
p.cModel.a = 20; 
p.cModel.b = 1/p.cModel.a;

[varsigma, tau, c, G] = gen_clutter_data(p, Nr);

subplot(1, 2, 2)
histogram(abs(varsigma(:)), 200, 'Normalization', 'pdf')

% overlay theoretical k-distribution curve
P = mean(abs(varsigma(:)).^2);
nu = p.cModel.a;
r = linspace(min(abs(varsigma(:))), max(abs(varsigma(:))), 200);

pdf_K = (4 * nu^((nu + 1)/2) * r.^(nu))/(gamma(nu) * P^((nu + 1)/2)) .* besselk(nu - 1, 2 * r * sqrt(nu/P));

hold on;
plot(r, pdf_K)

xlabel('Amplitude $r = |\varsigma|$', 'Interpreter', 'latex')
ylabel('$p(r)$', 'Interpreter', 'latex')
title(sprintf('$\\nu = %.1f$', p.cModel.a), 'Interpreter', 'latex')
legend('Simulated', 'K-distribution (theory)')

% divide each gate by sqrt(tau) to recover speckle and check Gaussian distributed
assert(p.cModel.Ktype == 3);
q_speckle = c ./ sqrt(tau);

%% Gaussian detectors in K-distributed clutter
p.cModel.Ktype = 3;
p.cModel.CNR = 20;
p.cModel.a = 1; p.cModel.b = 1/p.cModel.a;

Kt = 2 * NM;            % K_t training cells, K = 2NM (RMB)
Nr = Kt + 1;            % Nr = K + 1 gates

[varsigma, tau] = gen_clutter_data(p, Nr);
x_CUT = varsigma(:, 1);
training_cells = varsigma(:, 2:end);
Rt_hat = (training_cells * training_cells')/Kt;

% target space-time steering vector, s
nv = (0:p.radar.N - 1).';
mv = (0:p.radar.M - 1).';

theta_t = 0;
f_t = 0;

fs_t = (p.radar.d/p.radar.lambda) * sind(theta_t);      % target spatial frequency
s_sp = exp(1i * 2 * pi * fs_t * nv);                    % N x 1, phase across channels
s_temp = exp(1i * 2 * pi * f_t  * mv);                  % M x 1, phase across pulses
s = kron(s_temp, s_sp);                                 % NM x 1, same order as gen_clutter_data

% AMF, NAMF detector
w = Rt_hat \ s;
num = abs(w' * x_CUT)^2;
den_s = real(s' * w);
den_x = real(x_CUT' * (Rt_hat \ x_CUT));

y_AMF  = num / den_s;
y_NAMF = y_AMF / den_x;

%% Pfa in K-distributed clutter, thresholds set in Gaussian clutter
rng(3);
Pfa = 0.01;
Ntrials = 10000;

% set thresholds from Gaussian clutter
p.cModel.Ktype = 0;
[yAMF_G, yNAMF_G] = run_trials(p, Kt, s, Ntrials);
gamma_AMF = quantile(yAMF_G, 1 - Pfa);
gamma_NAMF = quantile(yNAMF_G, 1 - Pfa);

% apply the same thresholds in K-distributed clutter (fresh trials)
p.cModel.Ktype = 3;
p.cModel.a = 1; p.cModel.b = 1/p.cModel.a;
[yAMF_K, yNAMF_K] = run_trials(p, Kt, s, Ntrials);

Pfa_AMF = mean(yAMF_K > gamma_AMF);
Pfa_NAMF = mean(yNAMF_K > gamma_NAMF);

fprintf('designed Pfa = %.3f (+/- %.3f)\n', Pfa, 2*sqrt(Pfa*(1-Pfa)/Ntrials));
fprintf('AMF : actual Pfa = %.4f\n', Pfa_AMF);
fprintf('NAMF: actual Pfa = %.4f\n', Pfa_NAMF);

%% MVDR spectrum of true covariance
Pq = real(G(1, 1));                         % G(i, i) = Pq for every i
Pn = Pq * 10^(-p.cModel.CNR / 10);  

R = G + Pn * eye(size(G));                  % true R
R_inv = inv(R);

sin_theta = linspace(-1, 1, 201);
fd = linspace(-0.5, 0.5, 201);

P = zeros(length(fd), length(sin_theta));
nv = (0:p.radar.N - 1).';
mv = (0:p.radar.M - 1).';

for row = 1:length(fd)
    for col = 1:length(sin_theta)
        % target space-time steering vector, s
        fs_t = (p.radar.d/p.radar.lambda) * sin_theta(col);     % target spatial frequency
        s_sp = exp(1i * 2 * pi * fs_t * nv);                    % N x 1, phase across channels
        s_temp = exp(1i * 2 * pi * fd(row)  * mv);              % M x 1, phase across pulses
        s = kron(s_temp, s_sp);                                 % NM x 1, same order as gen_clutter_data

        P(row, col) = 1/real(s' * R_inv * s);
    end
end

figure
subplot(1, 2, 1)
imagesc(sin_theta, fd, 10*log10(NM * P / Pn))
axis xy
xlabel("sin\theta")
ylabel("f_d (cycles/pulse)")
ylabel(colorbar, "dB relative to P_n")
title("MVDR Spectrum, True Covariance R = G + P_nI (CNR 20dB)")

% alternatively, with steering angle theta on x-axis and 
theta_deg = linspace(-90, 90, 201);
P = zeros(length(fd), length(theta_deg));

for row = 1:length(fd)
    for col = 1:length(theta_deg)
        % target space-time steering vector, s
        fs_t = (p.radar.d/p.radar.lambda) * sind(theta_deg(col));     % target spatial frequency
        s_sp = exp(1i * 2 * pi * fs_t * nv);                          % N x 1, phase across channels
        s_temp = exp(1i * 2 * pi * fd(row)  * mv);                    % M x 1, phase across pulses
        s = kron(s_temp, s_sp);                                       % NM x 1, same order as gen_clutter_data

        P(row, col) = 1/real(s' * R_inv * s);
    end
end

subplot(1, 2, 2)
imagesc(theta_deg, fd * p.radar.PRF, 10*log10(NM * P / Pn))
axis xy; colorbar
xlabel('Steering angle (deg)'); ylabel('Frequency (Hz)')
ylabel(colorbar, 'dB relative to P_n')
title('MVDR Spectrum, True Covariance (angle-Hz axes)')

%% MVDR spectrum from the sample covariance (SCM) vs the true covariance
% Run after the parameter block of main.m (needs p and NM in the workspace).
% Gaussian training data (Ktype = 0), so the only error in R_hat is finite K.

p.cModel.Ktype = 0;
p.cModel.CNR = 20;
p.cModel.Dtype = 0;

Ks = [NM + 10, 2*NM, 10*NM];                % barely invertible, RMB rule, plenty

sin_theta = linspace(-1, 1, 201);
fd = linspace(-0.5, 0.5, 201);

% true covariance (G from the same call as the first training set)
[Z, ~, ~, G] = gen_clutter_data(p, Ks(1));
Pq = real(G(1, 1));
Pn = Pq * 10^(-p.cModel.CNR / 10);
R = G + Pn * eye(NM);

spectra = cell(1, 1 + length(Ks));
labels = cell(1, 1 + length(Ks));
spectra{1} = 10*log10(NM * mvdr_spectrum(R, p, asind(sin_theta), fd) / Pn);
labels{1} = 'True R = G + P_nI';

for k = 1:length(Ks)
    K = Ks(k);
    if k > 1
        Z = gen_clutter_data(p, K);         % fresh Gaussian training snapshots
    end
    R_hat = (Z * Z') / K;                   % sample covariance
    spectra{k + 1} = 10*log10(NM * mvdr_spectrum(R_hat, p, asind(sin_theta), fd) / Pn);
    labels{k + 1} = sprintf('SCM, K = %d', K);
end

% spectra side by side, same colour scale
clim_all = [min(cellfun(@(x) min(x(:)), spectra)), max(cellfun(@(x) max(x(:)), spectra))];
figure
for k = 1:length(spectra)
    subplot(2, 2, k)
    imagesc(sin_theta, fd, spectra{k}); axis xy
    clim(clim_all)                          % use caxis(clim_all) on MATLAB before R2022a
    xlabel('sin\theta'); ylabel('f_d (cycles/pulse)')
    ylabel(colorbar, 'dB relative to P_n')
    title(labels{k})
end

% cut through the spectrum at broadside (sin theta = 0), all on one axis
[~, c0] = min(abs(sin_theta));
figure; hold on
for k = 1:length(spectra)
    plot(fd, spectra{k}(:, c0), 'DisplayName', labels{k})
end
xlabel('f_d (cycles/pulse)'); ylabel('dB relative to P_n')
title('MVDR spectrum at broadside (sin\theta = 0)'); legend show; grid on

% noise floor: the median over the map (the ridge is a small fraction of it)
for k = 1:length(spectra)
    fprintf('%-20s floor = %6.2f dB\n', labels{k}, median(spectra{k}(:)));
end

%% save all figs
allFigures = allchild(groot); 

for i = 1:numel(allFigures)
    figHandle = allFigures(i);
    
    fileName = sprintf('fig%02d.png', figHandle.Number);
    fullPath = fullfile('figs', fileName); % Saves directly into the existing 'figs' directory
    
    exportgraphics(figHandle, fullPath, 'Resolution', 300);
end

%%
toc;