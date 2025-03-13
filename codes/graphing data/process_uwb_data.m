function process_uwb_data(data_file,len,set_range,range_lim,plot_frame_num,offset,fps,K,ovlap)
    % Process UWB radar data from a raw data file
    % Inputs:
    %   - data_file: Path to the .dat or .mat file containing the raw UWB data
    %   - len: Number of in-phase or quadrature samples per frame/number of
    %   range bins
    %   - set_range: the operating range of the uwb sensor set when
    %   recording data
    %   - range_lim: the range limit to be specified for plots
    %   - plot_frame_num: the number of radar frames to be plotted together
    %   - offset: the sample index offset for each radar frame
    %   - fps: frames per second recording speed of radar

    %% Load data
    dataFile = fopen(data_file,'rb');
    Data = fread(dataFile,'float');
    fclose(dataFile);
    %% Initialize parameters
    frame_size = 2 * len; % Total samples per frame (I + Q)

    % Total number of frames
    total_frames = floor((length(Data) - offset) / (frame_size + offset))
    if total_frames < 1
        error('Data length is insufficient for even a single frame with the given parameters.');
    end

    %% Process frames
    processed_frames = zeros(len, total_frames); % Preallocate space for processed data
    doppler_frames = zeros(len,total_frames);

    for frame_idx = 1:total_frames
        start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
        frame_I = Data(start_idx:start_idx + len - 1);
        frame_Q = Data(start_idx + len:start_idx + frame_size - 1);
        frame_complex = frame_I + 1i * frame_Q;
        processed_frames(:, frame_idx) = abs(frame_complex);
        doppler_frames(:, frame_idx) = frame_complex;
    end

% Initialize MTI filtered frames
mti_frames = zeros(len, total_frames - 1); % MTI will result in one less frame

processed_frames = rescale(processed_frames);
% MTI filter implementation
for frame_idx = 2:total_frames
    % Current frame indices
    start_idx_curr = offset + (frame_idx - 1) * (frame_size + offset) + 1;
    end_idx_curr = start_idx_curr + frame_size - 1;

    % Previous frame indices
    start_idx_prev = offset + (frame_idx - 2) * (frame_size + offset) + 1;
    end_idx_prev = start_idx_prev + frame_size - 1;

    if end_idx_curr > length(Data) || end_idx_prev > length(Data)
        warning('Skipping incomplete frame at the end of the dataset.');
        break;
    end

    frame_complex_curr = processed_frames(:, frame_idx);
    frame_complex_prev = processed_frames(:, frame_idx-1);

    % Rotate the previous frame by 180 degrees (negate complex conjugate)
    frame_complex_prev_rotated = -conj(frame_complex_prev);

    % Apply MTI filter: Add rotated previous frame to current frame
    frame_filtered = frame_complex_curr + frame_complex_prev_rotated;

    % Store the magnitude of the filtered frame
    mti_frames(:, frame_idx - 1) = abs(frame_filtered);
end

mti_doppler = zeros(len, total_frames); 

% MTI filter implementation for complex frames
for frame_idx = 2:total_frames
    frame_complex_curr = doppler_frames(:, frame_idx);
    frame_complex_prev = doppler_frames(:, frame_idx-1);
    frame_complex_prev_rotated = -(frame_complex_prev);
    frame_filtered = frame_complex_curr + frame_complex_prev_rotated;
    mti_doppler(:, frame_idx - 1) = frame_filtered;
end
%total_frames
%save('mti_doppler.mat','mti_doppler');

% Define the size of the moving average window for low-pass filtering
window_size = 5; 
half_window = floor(window_size / 2);

% Preallocate filtered frames
low_pass_frames = zeros(size(mti_frames));

% Apply moving average filter along the time dimension (Doppler filtering)
for range_idx = 1:len
    % Extract the time series for the current range bin
    time_series = mti_frames(range_idx, :);

    % Apply moving average filter
    for t = 1:size(mti_frames, 2)
        % Determine the indices for the moving average window
        start_idx = max(1, t - half_window);
        end_idx = min(size(mti_frames, 2), t + half_window);

        % Compute the moving average
        low_pass_frames(range_idx, t) = mean(time_series(start_idx:end_idx));
    end
end




       %% Apply FFT along the fast-time dimension (Range-FFT)
        %range_fft = fft(mti_frames, [], 1); % FFT along the fast-time dimension
        %range_fft = fft(doppler_frames,[],1);
        range_fft = mti_doppler;
        save('range_fft.mat','range_fft');
        %% Apply high-pass Butterworth filter to remove stationary clutter
        cutoff_freq = 0.02; % Cutoff frequency in normalized units
        [b, a] = butter(9, cutoff_freq, 'high'); % 9th-order high-pass Butterworth filter
        range_fft_filtered = filtfilt(b, a, range_fft);
        %range_fft_filtered = range_fft;

        %% Select range bins of interest (e.g., bins 5 to 20)
        range_bins_of_interest = 1:25; % Adjust based on your radar's range resolution
        range_fft_selected = range_fft_filtered(range_bins_of_interest, :);

        %% Apply STFT to extract micro-Doppler signatures
        %window_size = round(0.8 * fps); % 0.8-second Hamming window
        window_size = K;
        overlap = round(ovlap * window_size); % 95% overlap
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
    
    save('time_doppler_map.mat', 'time_doppler_map');
    %% Sum across range bins to generate the final time-Doppler map
    time_doppler_map_sum = sum(time_doppler_map, 3);
    save('time_doppler_map_sum.mat', 'time_doppler_map_sum');
    % Get global min/max (ADD THIS)
    min_doppler_sum = min(time_doppler_map_sum(:));
    max_doppler_sum = max(time_doppler_map_sum(:));
    fprintf('Dynamic Range: [%.2f, %.2f]\n', min_doppler_sum, max_doppler_sum);
    time_doppler_map_rescale = rescale(time_doppler_map_sum,"InputMin",0,"InputMax",1.5);
    %cutoff_freq = 0.0075; % Cutoff frequency in normalized units
    %[b, a] = butter(9, cutoff_freq, 'high'); % 9th-order high-pass Butterworth filter
    %time_doppler_map_rescale = filtfilt(b, a, time_doppler_map_rescale);
    %time_doppler_map_rescale = rescale(time_doppler_map_rescale);
    time_doppler_map_db = 20*log10(time_doppler_map_rescale);
    save('time_doppler_map_db.mat', 'time_doppler_map_db');

    low_pass_frames = rescale(low_pass_frames);
    %save('low_pass_frames.mat','low_pass_frames');
   
    %% Visualization
    % Convert sample index to range (in meters)
    range_per_sample = set_range / len; % Range corresponding to each sample
    range = (0:len-1) * range_per_sample; % Range vector
    total_time = total_frames / fps; % Total time in seconds
    time_vector = linspace(0, total_time, total_frames-1); % Time vector for x-axis (for MTI frames)

    %raw frames
    figure;
    hold on;
    for i = 1:min(plot_frame_num, total_frames) %plot the given amount of frames
        plot(range, processed_frames(:, i), 'DisplayName', ['Frame ', num2str(i)]);
    end
    hold off;
    %legend;
    xlim([0 range_lim]);
    title('Raw Radar Frames');
    xlabel('Range (m)');
    ylabel('Amplitude');

    %mti frames
    figure;
    hold on;
    for i = 1:min(plot_frame_num, total_frames-1) %plot the given amount of frames
        %plot(range, abs(mti_doppler(:, i)), 'DisplayName', ['Frame ', num2str(i)]);
        plot(range, low_pass_frames(:, i), 'DisplayName', ['Frame ', num2str(i)]);
    end
    hold off;
    %legend;
    xlim([0 range_lim]);
    title('MTI Filtered and Rescaled Radar Frames');
    xlabel('Range (m)');
    ylabel('Amplitude');
    

    %spectrogram
    figure;
    %pcolor(time_vector, range, 20 * log10(mti_frames));
    pcolor(time_vector, range, 20*log10(low_pass_frames));
    shading interp; 
    axis xy; % Ensure the origin is at the bottom-left
    colormap jet;
    colorbar;
    title('Radar Data Spectrogram');
    xlabel('Time (s)'); % Label x-axis as time in seconds
    ylabel('Range (m)');
    ylim([0 range_lim]); % Limit the y-axis to the specified range limit
    clim([-35 0]); % Adjust color limits for better visualization

    save('microdop.mat','time_doppler_map_db');
    %micro-doppler Spectrogram
    figure;
    imagesc(T,F,time_doppler_map_db)
    %pcolor(T,F,time_doppler_map_db);
    %shading interp;
    axis xy; % Ensure the origin is at the bottom-left
    colormap jet;
    colorbar;
    title('Micro-Doppler Signature');
    xlabel('Time (s)');
    ylabel('Doppler Frequency (Hz)');
    %ylim([-50 50]);
    clim([-60 0]); % Adjust color scale as needed

    %{
time_axis = linspace(0, 224, 224); % Time axis

% Define the Doppler frequency axis (in Hz)
doppler_axis = linspace(0, 224, 224); % Doppler frequency axis

% Plot the micro-Doppler signature
figure;
pcolor(time_axis, doppler_axis, time_doppler_map_db_224x224);
shading interp; % Smooth shading
axis xy; % Ensure the origin is at the bottom-left
colormap jet; % Use the jet colormap
colorbar; % Add a colorbar
title('224x224 Micro-Doppler Signature');
xlabel('Time (s)'); % Label x-axis as time in seconds
ylabel('Doppler Frequency (Hz)'); % Label y-axis as Doppler frequency in Hz
clim([-70 0]);
    %}
    %% range-doppler matrix (formatrack method)

%{
K = 128; % Doppler-FFT window size
num_doppler_bins = K; % Number of Doppler bins
window_function = hamming(K); % Apply windowing for Doppler-FFT

% Preallocate space for Range-Doppler matrix and time-Doppler spectrogram
range_doppler_matrices = zeros(len, num_doppler_bins, total_frames - K + 1);
time_doppler_map = zeros(num_doppler_bins, total_frames - K + 1);

%Apply low-pass Butterworth filter to remove stationary clutter
cutoff_freq = 0.4; % Cutoff frequency in normalized units
[b, a] = butter(9, cutoff_freq, 'low');
doppler_frames = filtfilt(b,a,doppler_frames);

% Loop through each radar frame
for frame_idx = 1:(total_frames - K - 1)
    % Extract the window of K frames
    doppler_window = doppler_frames(:, frame_idx:frame_idx + K - 1);
    
    % range-doppler fft
    doppler_spectrum = fftshift(fft(doppler_window .* window_function', K, 2),2);

    % Store the range-Doppler matrix
    range_doppler_matrices(:, :, frame_idx) = abs(doppler_spectrum);
    
    % Sum across range bins to form the time-Doppler map
    time_doppler_map(:, frame_idx) = sum(doppler_spectrum, 1);
    %time_doppler_map(:, frame_idx) = ifft(sum(doppler_spectrum, 1));
end

% Rescale the time-Doppler map for visualization
time_doppler_map = abs(time_doppler_map);
time_doppler_map_db = 20*log10(rescale(time_doppler_map));
%save('time_doppler_map_db.mat','time_doppler_map_db')

% Generate the Time-Doppler Spectrogram
time_axis = linspace(0, total_time, total_frames - K + 1);
doppler_axis = linspace(-num_doppler_bins/2, num_doppler_bins/2, num_doppler_bins);
%doppler_axis = linspace(-num_doppler_bins/2, num_doppler_bins/2, 1);

% Plot the Time-Doppler Spectrogram
figure;
pcolor(time_axis, doppler_axis, time_doppler_map_db);
shading interp; 
%imagesc(time_axis, doppler_axis, time_doppler_map_db);
axis xy; % Flip axes to keep the origin bottom-left
colormap jet;
colorbar;
title('micro-doppler signature');
xlabel('time');
ylabel('doppler bin');
ylim([-20, 20]);
clim([-80, 0]); % Adjust color scale as needed
%}

end


    %% Save processed frames
    %save('processed_uwb_frames.mat', 'processed_frames');
    %disp('Processed frames saved to processed_uwb_frames.mat');
%end