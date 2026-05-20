% FOFsyn1formant.m — Visualize one FOF burst in time and frequency domain
%
% Uses the updated FOFburst with independent beta and dynamic T60 length.

% Parameters
fs = 44100;          % Sampling rate
Fc = 753;            % Formant center frequency (Hz)
BW = 100;            % Bandwidth (Hz)
A = 1.0;             % Amplitude

% Generate FOF burst using updated function
fof = FOFburst(Fc, BW, A, fs);

% Time vector in ms
t = (0:length(fof)-1) / fs * 1000;

% Plot: Time-domain
figure;
subplot(2,1,1);
plot(t, fof, 'b');
xlabel('Time (ms)');
ylabel('Amplitude');
title(['FOF Burst (Time Domain) — Fc = ', num2str(Fc), ' Hz, BW = ', num2str(BW), ' Hz']);
xlim([0 40]);

% Plot: Frequency-domain (zero-padding, no Hann window)
NFFT = 2^(nextpow2(length(fof)) + 2);  % extra zero-padding
S = abs(fft(fof, NFFT));
f_axis = fs * (0:(NFFT/2 - 1)) / NFFT;

subplot(2,1,2);
plot(f_axis, 20*log10(S(1:NFFT/2) + 1e-12), 'k');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('FOF Burst (Frequency Domain)');
xlim([0 5000]);

% Mark formant center frequency
xline(Fc, 'r--');
text(Fc + 200, max(20*log10(S(1:NFFT/2))) - 20, ...
    ['Fc = ', num2str(Fc), ' Hz'], ...
    'Color', 'red', 'FontSize', 10);
