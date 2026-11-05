function [y_AMF, y_NAMF, y_CLV, y_OKD] = run_trials(p, Kt, s, alpha, Ntrials, G, Pn)

% run_trials_Pd():
% runs Monte Carlo trials of the AMF, NAMF and OKD detectors, returning
% test statistics
%
% inputs: p: parameter struct (as for gen_clutter_data), with Ktype = 3
%         Kt: number of training cells
%         s: NM x 1 target steering vector
%         Ntrials: number of independent trials
%         G, Pn: true speckle covariance and noise power (from
%         gen_clutter_data)
% returns: y_AMF, y_NAMF, y_CLV, y_OKD, y_TEN: Ntrials x 1 test statistics

NM = length(s);
Nr = Kt + 1;

% G = (G + G')/2;     % enforce exact Hermitian (float-point safety for eig below)

y_AMF = zeros(Ntrials, 1);
y_NAMF = zeros(Ntrials, 1);

want_CLV = (nargout >= 3);
want_OKD = (nargout >= 4);

if want_CLV
    y_CLV = zeros(Ntrials, 1);
    G = (G + G')/2;
end 

if want_OKD
    y_OKD = zeros(Ntrials,1);
    G = (G + G')/2;

    % precompute OKD quantities (fixed across trials)
    [U, Lam] = eig(G, 'vector');                 % G = U*diag(Lam)*U'
    Lam = real(Lam);                             % L x 1 eigenvalues
    s_t = U' * s;                                % steering vector in G's eigenbasis
    a = p.cModel.a;  bb = p.cModel.b;            % texture law (b renamed to avoid clash)
    
    % texture grid covering the Gamma mass (adapts to a,b)
    tg = linspace(gaminv(5e-4, a, bb), gaminv(0.9999, a, bb), 150);   % 1 x T
    dt = tg(2) - tg(1);
    logw = (a-1)*log(tg) - tg/bb - gammaln(a) - a*log(bb) + log(dt);  % log[Gamma pdf * d(tau)]
end

for n = 1:Ntrials   
    [varsigma, tau] = gen_clutter_data(p, Nr);
    
    x_CUT = varsigma(:, 1);
    if alpha ~= 0
        x_CUT = x_CUT + alpha * s;
    end
    training_cells = varsigma(:, 2:end);
    Rt_hat = (training_cells * training_cells') / Kt;

    % adaptive detectors
    w = Rt_hat \ s;
    num = abs(w' * x_CUT)^2;
    den_s = real(s' * w);
    den_x = real(x_CUT' * (Rt_hat \ x_CUT));

    y_AMF(n) = num / den_s;
    y_NAMF(n) = num / (den_s * den_x);

    % clairvoyant (given tau, then conditionally CN dist => apply WMF)
    if want_CLV
        Rc_inv = (tau(1) * G + Pn * eye(NM)) \ eye(NM);
        y_CLV(n) = abs(s' * Rc_inv * x_CUT)^2 / real(s' * Rc_inv * s);
    end

    % OKD: texture-marginalised optimum (GLRT in amplitude) ---
    if want_OKD
        xt   = U' * x_CUT;                        % L x 1  data in eigenbasis
        D    = Lam*tg + Pn;                       % L x T  = (tau*lambda_i + Pn)
        sRix = sum(conj(s_t).*xt ./ D, 1);        % 1 x T  s^H R_tau^-1 x
        sRis = sum(abs(s_t).^2 ./ D, 1);          % 1 x T  s^H R_tau^-1 s
        xRix = sum(abs(xt).^2 ./ D, 1);           % 1 x T  x^H R_tau^-1 x
        q1   = xRix - abs(sRix).^2 ./ sRis;       % H1 quadratic form (alpha profiled out)
        logdet = sum(log(D), 1);                  % 1 x T  log det R_tau
        l0 = -logdet - xRix + logw;               % log integrand, H0
        l1 = -logdet - q1   + logw;               % log integrand, H1
        y_OKD(n) = logsumexp_row(l1) - logsumexp_row(l0);   % log-likelihood ratio
    end
end

end

%% helper function
function y = logsumexp_row(v)
% numerically stable log(sum(exp(v))) for a row vector v
m = max(v);
y = m + log(sum(exp(v - m)));
end