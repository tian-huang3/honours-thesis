function varsigma = gen_clutter_data(p, Nr)

% gen_clutter_data(): 
% - function that returns Nr x clutter + noise snapshots (of length NM). 
% - angles are broadside angles theta (0 = broadside, +-90 deg = along the array axis / direction of flight)
% 
% inputs: p: structure that contains two subfields, radar and cModel
%           p.radar - structure that contains the following radar parameters
%               p.radar.N: number of spatial channels
%               p.radar.M: number of pulses
%               p.radar.PRF: pulse repetition frequency (Hz)
%               p.radar.lambda: wavelength (m)
%               p.radar.vp: platform velocity (m/s)
%               p.radar.d: inter-element spacing (m)
%               p.radar.thetaL: Look angle of the array (deg)
%           p.cModel - structure containing the clutter parameters and options
%               p.cModel.Ntheta: Number of clutter patches
%               p.cModel.Ktype: Model type (0 - standard, 1 - same texture applied to all azimuth angles, 2 - texture varies with azimuth)
%               p.cModel.Dtype: 0 - no clutter motion, 1 - clutter motion modelled as Gaussian
%               p.cModel.a: Shape parameter
%               p.cModel.b: scale parameter
%               p.cModel.u: variance of normalised Doppler ((cycles/pulse)^2)
%               p.cModel.CNR: clutter to noise ratio (dB)
%         Nr: Number of snapshots to generate/number of range gates

%% set parameters
N = p.radar.N;
M = p.radar.M;
Ntheta = p.cModel.Ntheta;
thetaL = pi * p.radar.thetaL / 180;
vp = p.radar.vp;
lambda = p.radar.lambda;
d = p.radar.d;
PRF = p.radar.PRF;

a = p.cModel.a;
b = p.cModel.b;
CNR = p.cModel.CNR;
Ktype = p.cModel.Ktype;
Dtype = p.cModel.Dtype;
u = p.cModel.u;

NM = N * M;
nv = (0:N - 1).';
mv = (0:M - 1).';

%% patch geometry around iso-range ring
theta = linspace(-pi/2, pi/2, Ntheta);
dtheta = theta(2) - theta(1);

fs = (d/lambda) * sin(theta);                  % spatial freq of each patch
fd = (2*vp/lambda) * (sin(theta)/PRF);         % Doppler freq of each patch

if Dtype == 1
    fd = fd + sqrt(u)*randn(size(fd));       % in the case of Gaussian clutter motion
end

%% antenna directivity per patch
DG = transpose((0.5 * (1 + cos(2*(theta-thetaL)))).^2);

%% space-time steering vector per patch
S = zeros(NM, Ntheta);

for k = 1:Ntheta
    s_sp = exp(1i * 2 * pi * fs(k) * nv);
    s_temp = exp(1i * 2 * pi * fd(k) * mv);
    S(:, k) = kron(s_temp, s_sp);
end

%% texture 
if Ktype == 0                   
    sigma_c = 1;                                            % standard
elseif Ktype == 1               
    sigma_c = sqrt(gamrnd(a, b, 1, 1) * ones(Ntheta, 1));   % same texture applied to all azimuth angles
elseif Ktype == 2               
    sigma_c = sqrt(gamrnd(a, b, Ntheta, 1));                % texture varies with azimuth
end

%% sum over patches (Melvin (13))
Av = (sigma_c / sqrt(2)) .* (randn(Ntheta, Nr) + 1i * randn(Ntheta, Nr));
c = dtheta * S * (Av .* DG);

%% add thermal noise at specified CNR (c + n)
Pc = mean(abs(c(:)).^2);

Pn = (Pc) * 10^(-CNR/10);
sigma_n = sqrt(Pn/2);

varsigma = c + sigma_n * (randn(NM, Nr) + 1i*randn(NM, Nr));

end