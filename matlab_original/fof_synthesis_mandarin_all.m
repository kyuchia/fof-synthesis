% fof_synthesis_mandarin_all.m — FOF Synthesis for Mandarin Vowels
%
% Fixed version:
%   - Calls FOFburst() instead of duplicating FOF logic inline
%   - Amplitudes in dB (converted to linear), matching CHANT conventions
%   - Dynamic burst length via T60 criterion (from FOFburst)
%   - Beta decoupled from Fc (default ~3 ms attack)

fs = 44100;
f0 = 220;
T = round(fs / f0);
duration = 1;
N = fs * duration;

% Vowel labels, formants [F1 F2 F3], bandwidths, amplitudes in dB
vowels = {
    'a',  [753, 1298, 3068], [100, 120, 150], [0, -7, -12];
    'i',  [309, 2416, 3438], [90,  110, 130], [0, -9, -16];
    'u',  [349,  759, 2883], [80,  100, 120], [0, -9, -16];
    'o',  [505,  856, 3052], [90,  110, 130], [0, -7, -12];
    'ei', [460, 2453, 3168], [100, 120, 150], [0, -7, -12];
};

for v = 1:size(vowels, 1)
    vowel_label = vowels{v, 1};
    F   = vowels{v, 2};
    BWs = vowels{v, 3};
    amps_dB = vowels{v, 4};

    % Convert dB to linear amplitude
    Amps = 10.^(amps_dB / 20);

    signal = zeros(1, N);

    for t = 1:T:N
        for i = 1:3
            % Call FOFburst — single source of truth
            burst = FOFburst(F(i), BWs(i), Amps(i), fs);
            burst_len = length(burst);

            % Handle end-of-signal boundary
            end_idx = min(t + burst_len - 1, N);
            actual_len = end_idx - t + 1;
            signal(t:end_idx) = signal(t:end_idx) + burst(1:actual_len);
        end
    end

    % Normalize
    signal = signal / max(abs(signal));

    % Save audio
    filename = ['fof_mandarin_' vowel_label '.wav'];
    audiowrite(filename, signal, fs);

    % Plot and export figure
    t_axis = (0:length(signal)-1) / fs;
    fig = figure('Name', ['/' vowel_label '/'], 'Visible', 'off');

    % Time-domain plot
    subplot(2, 1, 1);
    plot(t_axis, signal);
    title(['FOF Synthesized Mandarin /' vowel_label '/']);
    xlabel('Time (s)'); ylabel('Amplitude');

    % Spectrogram
    subplot(2, 1, 2);
    spectrogram(signal, hamming(1024), 768, 1024, fs, 'yaxis');
    title(['Spectrogram of /' vowel_label '/']);
    ylim([0 5]);  % show up to 5 kHz

    % Export plot
    exportgraphics(fig, ['fof_' vowel_label '_plot.png'], 'Resolution', 300);
    close(fig);
end

fprintf('Synthesis complete for all vowels.\n');
