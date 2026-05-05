% Multi-vowel FOF Synthesis for Mandarin Vowels (Enhanced Version)
fs = 44100;
f0 = 220;
T = round(fs / f0);
duration = 1; 
N = fs * duration;
fof_len = round(0.1 * fs);

% Vowel labels and formants: [F1 F2 F3]
vowels = {
    'a',  [753, 1298, 3068], [100, 120, 150], [1.0, 0.8, 0.5];
    'i',  [309, 2416, 3438], [90,  110, 130], [1.0, 0.7, 0.4];
    'u',  [349,  759, 2883], [80,  100, 120], [1.0, 0.7, 0.4];
    'o',  [505,  856, 3052], [90,  110, 130], [1.0, 0.7, 0.5];
    'ei', [460, 2453, 3168], [100, 120, 150], [1.0, 0.8, 0.5];
};

for v = 1:size(vowels,1)
    vowel_label = vowels{v,1};
    F = vowels{v,2};
    BWs = vowels{v,3};
    Amps = vowels{v,4};
    
    signal = zeros(1, N);
    
    for t = 1:T:(N - fof_len)
        for i = 1:3
            Fc = F(i);
            BW = BWs(i);
            A = Amps(i);
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
            signal(t:(t + fof_len - 1)) = signal(t:(t + fof_len - 1)) + fof;
        end
    end
    
    % Normalize
    signal = signal / max(abs(signal));

    % Save audio
    filename = ['fof_mandarin_' vowel_label '.wav'];
    audiowrite(filename, signal, fs);

    % Plot and export figure
    t_axis = (0:length(signal)-1)/fs;
    fig = figure('Name', ['/' vowel_label '/'], 'Visible', 'off');
    
    % Time-domain plot
    subplot(2,1,1);
    plot(t_axis, signal);
    title(['FOF Synthesized Mandarin /' vowel_label '/']);
    xlabel('Time (s)'); ylabel('Amplitude');

    % Spectrogram (instead of FFT)
    subplot(2,1,2);
    spectrogram(signal, hamming(1024), 768, 1024, fs, 'yaxis');
    title(['Spectrogram of /' vowel_label '/']);
    ylim([0 5]) % show up to 5kHz

    % Export plot
    exportgraphics(fig, ['fof_' vowel_label '_plot.png'], 'Resolution', 300);
    close(fig); % close figure to avoid clutter
end
