% FOFsyn1formant.m — Visualize one FOF burst in time and frequency domain

% Parameters
fs = 44100;          % Sampling rate
Fc = 753;            % Formant center frequency (Hz)
BW = 100;            % Bandwidth (Hz)
A = 1.0;             % Amplitude

% Generate FOF burst using your custom function
fof = FOFburst(Fc, BW, A, fs);

% Time vector in ms
t = (0:length(fof)-1) / fs * 1000;

% Plot: Time-domain
figure;
subplot(2,1,1);
plot(t, fof, 'b');
xlabel('Time (ms)');
ylabel('Amplitude');
title(['FOF Burst (Time Domain) — Fc = ', num2str(Fc), ' Hz']);
xlim([0 40]);

% Plot: Frequency-domain
NFFT = 2^nextpow2(length(fof));
S = abs(fft(fof .* hann(length(fof))', NFFT));
f_axis = fs * (0:(NFFT/2 - 1)) / NFFT;

subplot(2,1,2);
plot(f_axis, 20*log10(S(1:NFFT/2)), 'k');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('FOF Burst (Frequency Domain)');
xlim([0 5000]);

% Add red vertical line and horizontal label for Fc
xline(Fc, 'r--');
text(Fc + 260, max(20*log10(S)) - 25, ...
    ['F1 = ', num2str(Fc), ' Hz'], ...
    'Color', 'red', 'FontSize', 10);
