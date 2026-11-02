function [y_AMF, y_NAMF] = run_trials(p, Kt, s, Ntrials)

% run_trials():
% - Monte Carlo H0 (no target) trials of the AMF and NAMF
%
% inputs: p: parameter struct (as for gen_clutter_data)
%         Kt: number of training cells
%         s: NM x 1 target steering vector
%         Ntrials: number of independent trials
% returns: y_AMF, y_NAMF: Ntrials x 1 test statistics

Nr = Kt + 1;
y_AMF = zeros(Ntrials, 1);
y_NAMF = zeros(Ntrials, 1);

for n = 1:Ntrials
    varsigma = gen_clutter_data(p, Nr);

    x_CUT = varsigma(:, 1);
    training_cells = varsigma(:, 2:end);
    Rt_hat = (training_cells * training_cells')/Kt;

    w = Rt_hat \ s;
    num = abs(w' * x_CUT)^2;
    den_s = real(s' * w);
    den_x = real(x_CUT' * (Rt_hat \ x_CUT));

    y_AMF(n) = num / den_s;
    y_NAMF(n) = num / (den_s * den_x);
end

end