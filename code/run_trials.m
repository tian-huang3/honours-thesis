function [y_AMF, y_NAMF, y_OKD] = run_trials_Pd(p, Kt, s, alpha, Ntrials, G, Pn)

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
% returns: y_AMF, y_NAMF, y_OKD: Ntrials x 1 test statistics

NM = length(s);
Nr = Kt + 1;

y_AMF = zeros(Ntrials, 1);
y_NAMF = zeros(Ntrials, 1);

want_OKD = (nargout >= 3);
if want_OKD
    y_OKD = zeros(Ntrials, 1);
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
    
    % OKD (clairvoyant - interference covariance of CUT is taken as known)
    if want_OKD
        Rc_inv = (tau(1) * G + Pn * eye(NM)) \ eye(NM);
        y_OKD(n) = abs(s' * Rc_inv * x_CUT)^2 / real(s' * Rc_inv * s);
    end
end

end