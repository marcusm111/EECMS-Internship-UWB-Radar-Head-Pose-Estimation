function process(data_file, len, offset, fps)
    % Process radar data from a raw data file
    % Inputs:
    %   - data_file: Path to the .dat or .mat file containing the raw radar data
    %   - len: Number of time samples per sweep (128 in this case)
    %   - set_range: Operating range of the radar sensor
    %   - range_lim: Range limit for plots
    %   - plot_frame_num: Number of radar frames to plot
    %   - offset: Sample index offset for each radar frame
    %   - fps: Frames per second recording speed of radar

    %% Load data
    dataFile = fopen(data_file, 'rb');
    Data = fread(dataFile, 'float');
    fclose(dataFile);

    %% Extract in-phase (I) and quadrature (Q) data and combine into complex frames
    frame_size = 2 * len; % Total samples per frame (I + Q)
    total_frames = floor((length(Data) - offset) / (frame_size + offset));
    processed_frames = zeros(len, total_frames); % Preallocate space for processed data

    for frame_idx = 1:total_frames
        start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
        frame_I = Data(start_idx:start_idx + len - 1);
        frame_Q = Data(start_idx + len:start_idx + frame_size - 1);
        frame_complex = frame_I + 1i * frame_Q; % Combine I and Q into complex frame
        processed_frames(:, frame_idx) = frame_complex; % Store complex frame
    end

    %% Apply FFT along the fast-time dimension (Range-FFT)
    range_fft = fft(processed_frames, [], 1); % FFT along the fast-time dimension

    %% Apply high-pass Butterworth filter to remove stationary clutter
    cutoff_freq = 0.0075; % Cutoff frequency in normalized units
    [b, a] = butter(9, cutoff_freq, 'high'); % 9th-order high-pass Butterworth filter
    range_fft_filtered = filtfilt(b, a, range_fft); % Apply filter along the slow-time dimension

    %% Select range bins of interest (e.g., bins 5 to 25)
    range_bins_of_interest = 5:20; % Adjust based on your radar's range resolution
    range_fft_selected = range_fft_filtered(range_bins_of_interest, :);

    %% Apply STFT to extract micro-Doppler signatures
    window_size = round(0.8 * fps);
    overlap = round(0.5 * window_size); % 50% overlap
    hamming_window = hamming(window_size);

    % Preallocate space for STFT output
    num_range_bins_selected = length(range_bins_of_interest);
    num_time_frames = floor((total_frames - window_size) / (window_size - overlap)) + 1;
    time_doppler_map = zeros(window_size, num_time_frames, num_range_bins_selected);

    % Perform STFT for each selected range bin
    for range_idx = 1:num_range_bins_selected
        time_series = range_fft_selected(range_idx, :);
        [S, F, T] = stft(time_series, fps, 'Window', hamming_window, 'OverlapLength', overlap, 'FFTLength', window_size);
        time_doppler_map(:, :, range_idx) = abs(S); % Store magnitude of STFT
    end

    %% Sum across range bins to generate the final time-Doppler map
    time_doppler_map_sum = sum(time_doppler_map, 3);

    %% Convert to dB scale
    time_doppler_map_db = 20 * log10(rescale(time_doppler_map_sum));

    %% Generate time and frequency axes
    time_axis = T; % Time axis from STFT
    frequency_axis = F; % Frequency axis from STFT

    %% Plot the Micro-Doppler Spectrogram
    figure;
    pcolor(time_axis, frequency_axis, time_doppler_map_db);
    shading interp;
    axis xy; % Ensure the origin is at the bottom-left
    colormap jet;
    colorbar;
    title('Micro-Doppler Signature');
    xlabel('Time (s)');
    ylabel('Doppler Frequency (Hz)');
    ylim([-50 50]);
    clim([-60, 0]); % Adjust color scale as needed

    %% Save processed data
    save('time_doppler_map_db.mat', 'time_doppler_map_db');
    disp('Micro-Doppler signature saved to time_doppler_map_db.mat');
end

 %Apply FFT along the fast-time dimension (Range-FFT)
    range_fft = fft(doppler_frames, [], 1); % FFT along the fast-time dimension

    %Apply high-pass Butterworth filter to remove stationary clutter
    cutoff_freq = 0.0075; % Cutoff frequency in normalized units
    [b, a] = butter(9, cutoff_freq, 'high'); % 9th-order high-pass Butterworth filter
    range_fft_filtered = filtfilt(b,a,range_fft);
    save('range_fft_filtered.mat','range_fft_filtered');

    %Select range bins of interest (e.g., bins 5 to 25)
    range_bins_of_interest = 5:20; % Adjust based on your radar's range resolution
    range_fft_selected = range_fft_filtered(range_bins_of_interest, :);

    %Apply STFT to extract micro-Doppler signatures
    window_size = round(0.8 * fps); % 0.2-second Hamming window
    overlap = round(0.95 * window_size); % 95% overlap
    hamming_window = hamming(window_size);

    % Preallocate space for STFT output
    num_range_bins_selected = length(range_bins_of_interest);
    num_time_frames = floor((total_frames - window_size) / (window_size - overlap)) + 1;
    time_doppler_map = zeros(window_size, num_time_frames, num_range_bins_selected);

    % Perform STFT for each selected range bin
    for range_idx = 1:num_range_bins_selected
        time_series = range_fft_selected(range_idx, :);
        [S, F, T] = stft(time_series, fps, 'Window', hamming_window, 'OverlapLength', overlap, 'FFTLength', window_size);
        time_doppler_map(:, :, range_idx) = abs(S);
    end

    %Sum across range bins to generate the final time-Doppler map
    time_doppler_map_sum = sum(time_doppler_map, 3);
    time_doppler_map_db = 20*log10(rescale(time_doppler_map_sum));
