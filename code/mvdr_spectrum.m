function P = mvdr_spectrum(R, p, theta_deg, fd)

% mvdr_spectrum():
% computes MVDR (optimum) angle-Doppler spectrum P = 1/(s^H R^-1 s), scanned over a grid
%
% inputs: R: NM x NM covariance (true or estimated)
%         p: parameter struct (as for gen_clutter_data), uses p.radar.N, M, d, lambda
%            theta_deg: 1 x Ntheta look angles (deg, broadside = 0)
%         fd: 1 x Nfd normalised Doppler (cycles/pulse)
% returns: P: Nfd x Ntheta spectrum (rows = Doppler, cols = angle, ready for imagesc)

nv = (0:p.radar.N - 1).';
mv = (0:p.radar.M - 1).';
R_inv = inv(R);

fs = (p.radar.d/p.radar.lambda) * sind(theta_deg);     % 1 x Ntheta spatial freqs
S_sp = exp(1i * 2 * pi * nv * fs);                      % N x Ntheta, one column per look angle

P = zeros(length(fd), length(theta_deg));

for row = 1:length(fd)
    s_temp = exp(1i * 2 * pi * fd(row) * mv);           % M x 1
    S = kron(s_temp, S_sp);                             % NM x Ntheta, every angle at this Doppler
    P(row, :) = 1 ./ real(sum(conj(S) .* (R_inv * S), 1));   % s^H R^-1 s for each column at once
end

end
