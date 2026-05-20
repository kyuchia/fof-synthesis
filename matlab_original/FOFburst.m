function fof = FOFburst(Fc, BW, A, fs, beta)
% FOFburst Generate a single FOF burst for a formant.
%
%   fof = FOFburst(Fc, BW, A, fs)
%   fof = FOFburst(Fc, BW, A, fs, beta)
%
%   Fc   = formant center frequency (Hz)
%   BW   = formant bandwidth (Hz)
%   A    = amplitude (linear)
%   fs   = sampling rate (Hz)
%   beta = attack rate (rad/sample), controls skirt width independently
%          of bandwidth. Default: pi / (0.003 * fs) (~3 ms attack).
%
%   Based on Rodet (1984), "Time-Domain Formant-Wave-Function Synthesis."

% Default beta: ~3 ms attack (independent of Fc)
if nargin < 5
    beta = pi / (0.003 * fs);
end

% Derived parameters
alpha = pi * BW / fs;           % exponential decay rate
omega_c = 2 * pi * Fc / fs;    % angular center frequency

% Dynamic burst length based on T60 decay criterion
t60 = 6.91 / (pi * BW);        % T60 in seconds
fof_len = max(ceil(t60 * fs), 64);  % minimum 64 samples

% Attack duration in samples
k_a = round(pi / beta);

% Vectorized computation
k = 0:(fof_len - 1);
env = exp(-alpha * k);
carrier = sin(omega_c * k);
attack = 0.5 * (1 - cos(beta * k));
attack(k > k_a) = 1;

fof = A * attack .* env .* carrier;
end
