function fof = FOFburst(Fc, BW, A, fs)
% FOFburst Generate a single FOF burst for a formant.
%   fof = FOFburst(Fc, BW, A, fs)
%   Fc = formant center frequency (Hz)
%   BW = formant bandwidth (Hz)
%   A  = amplitude
%   fs = sampling rate (Hz)

% Duration (100 ms by T60 rule)
fof_len = round(0.1 * fs);
alpha = pi * BW / fs;
omega_c = 2 * pi * Fc / fs;
beta = omega_c / 10;
phi = 0;
k_a = round(pi / beta);
fof = zeros(1, fof_len);

for k = 1:fof_len
    env = exp(-alpha * (k - 1));
    sin_part = sin(omega_c * (k - 1) + phi);
    if k <= k_a
        fof(k) = A * 0.5 * (1 - cos(beta * (k - 1))) * env * sin_part;
    else
        fof(k) = A * env * sin_part;
    end
end
end
