function [varsigma, tau, c, G, R, Pn, Pq, beta] = gen_clutter_data(p, Nr)

% gen_clutter_data():
% function that returns Nr x clutter + noise snapshots (of length L = NM). 
% 
% inputs: p: structure that contains two subfields, radar and cModel
%           p.radar - structure that contains the following radar parameters
%               p.radar.N: number of spatial channels
%               p.radar.M: number of pulses
%               p.radar.PRF: pulse repetition frequency (Hz)
%               p.radar.lambda: wavelength (m)
%               p.radar.vp: platform velocity (m/s)
%               p.radar.d: inter-element spacing (m)
%               p.radar.thetaL: look angle of the array (deg) (direction the
%               antennas main beam points, thetaL = 0 => beam perpendicular to
%               flight path)
%           p.cModel - structure containing the clutter parameters and options
%               p.cModel.Ntheta: Number of clutter patches
%               p.cModel.Ktype: Model type
%                   0 - standard (fixed tau = 1) for all azimuth angles &
%                   range gates (Gaussian clutter)
%                   1 - same texture (from Gamma dist.) applied to all
%                   azimuth angles (conditioned on tau, is complex
%                   Gaussian)
%                   2 - texture varies with azimuth, (by CLT, textures avg
%                   out in coherent sum => Gaussian)
%                   3 - different texture for each range cell
%                   (K-distributed)
%               p.cModel.Dtype: 
%                   0 - no clutter motion
%                   1 - clutter motion modelled as Gaussian
%               p.cModel.a: Shape parameter
%               p.cModel.b: scale parameter
%               p.cModel.u: variance of normalised Doppler ((cycles/pulse)^2)
%               p.cModel.CNR: clutter to noise ratio (dB)
%         Nr: Number of snapshots to generate (number of range gates)
% returns: varsigma: L (= NM) x Nr clutter + noise snapshots
%          tau: texture actually applied (scalar, Ntheta x 1 or 1 x Nr depending on Ktype)
%          c: clutter snapshots
%          G: speckle covariance matrix
%          R: (clutter + noise) covariance matrix
%          Pn: noise power
%          Pq: speckle power
%          beta: clutter ridge slope
% notes:
% - Angles are broadside angles theta (0 = broadside, +-90 deg = along the array axis / direction of flight)

%% set parameters
N = p.radar.N;
M = p.radar.M;
Ntheta = p.cModel.Ntheta;
thetaL = p.radar.thetaL * (pi/180);          % convert deg to rad
vp = p.radar.vp;
lambda = p.radar.lambda;
d = p.radar.d;
PRF = p.radar.PRF;

a = p.cModel.a;    % bigger a = more homogeneous, smaller a = more heterogeneous
b = p.cModel.b; 
CNR = p.cModel.CNR;
Ktype = p.cModel.Ktype;
Dtype = p.cModel.Dtype;
u = p.cModel.u;

NM = N * M;

%% patch geometry around iso-range ring
theta = linspace(-pi/2, pi/2, Ntheta);      % theta (rad)
dtheta = theta(2) - theta(1);               % later, the sum over patches is a Riemann 
                                            % sum approximating Melvin's integral over the ring, 
                                            % so dtheta is multiplied

fs = (d/lambda) * sin(theta);               % spatial freq of each patch
fd = (2*vp/lambda) * (sin(theta)/PRF);      % Doppler freq of each patch (note relative radial velocity 
                                            % between radar and scatterer is just platform velocity in 
                                            % this case, since scatterer is stationary clutter)
                                       
beta = (2 * vp)/(PRF * d);                  % clutter ridge slope

if Dtype == 1                               % TO ASK: not used
    fd = fd + sqrt(u)*randn(size(fd));      % in the case of Gaussian clutter motion
end

%% antenna directivity per patch
DG = transpose((0.5 * (1 + cos(2*(theta-thetaL)))).^2);

%% space-time steering vector per patch
nv = (0:N - 1).';
mv = (0:M - 1).';

SSc = zeros(NM, Ntheta);                     % L x Ntheta patch steering matrix (each column is the 
                                            % N x M (=L) spatio-temporal steering matrix for each patch)
for patch = 1:Ntheta
    s_sp = exp(1i * 2 * pi * fs(patch) * nv);
    s_temp = exp(1i * 2 * pi * fd(patch) * mv);
    SSc(:, patch) = kron(s_temp, s_sp);
end

%% texture, sigma_c = sqrt(tau) = std of clutter given texture 
if Ktype == 0                   
    sigma_c = 1;                                            % standard (fixed tau = 1) for all azimuth & range gates
elseif Ktype == 1               
    sigma_c = sqrt(gamrnd(a, b, 1, 1) * ones(Ntheta, 1));   % fixed random (gamma dist.) tau for all azimuth & range gates
elseif Ktype == 2               
    sigma_c = sqrt(gamrnd(a, b, Ntheta, 1));                % texture varies with azimuth which is reused for each range gate
elseif Ktype == 3
    sigma_c = sqrt(gamrnd(a, b, 1, Nr));                    % same common texture across patches (azimuth), but different for each range gate
end    
tau = sigma_c.^2;

%% sum over patches (Melvin eq (13))
Av = (sigma_c / sqrt(2)) .* (randn(Ntheta, Nr) + 1i * randn(Ntheta, Nr));
c = dtheta * SSc * (Av .* DG);              % weights each patch by antenna gain DG
G = dtheta^2 * SSc * diag(DG.^2) * SSc';    % speckle covariance (tau = 1, no noise)

%% add thermal noise at specified CNR (c + n)
Pq = dtheta^2 * sum(DG.^2);

Pn = (Pq) * 10^(-CNR/10);
sigma_n = sqrt(Pn/2);

varsigma = c + sigma_n * (randn(NM, Nr) + 1i*randn(NM, Nr));

R = G + Pn*eye(NM);

end