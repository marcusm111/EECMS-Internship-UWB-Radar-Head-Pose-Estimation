function five_pos_microdop_sig(test_set,name,data_up,data_down,data_left,data_right,data_nomovement,K,ovlap,len,offset,fps)
    % Process UWB radar data from five raw data files and visualize their micro-Doppler signatures
    % Inputs: 
    %   - test_set: Test set number
    %   - name: Radar subject name
    %   - data_up,data_down,data_left,data_right,data_nomovement: recorded head pose
    %   radar data sets
    %   - K: STFT window size
    %   - ovlap: STFT window overlapping factor
    %   - len: number of range bins
    %   - offset: radar data sample offset
    %   - fps: radar recording fps

    data_files = {data_up, data_down, data_left, data_right, data_nomovement};
    labels = {'Up', 'Down', 'Left90', 'Right90', 'No Movement'};
    num_positions = numel(data_files);
    processed_spectrograms = cell(1, num_positions);
    time_vectors = cell(1, num_positions);
    frequency_vectors = cell(1, num_positions);
    figure;

    for pos_idx = 1:num_positions
        %% Load data
        dataFile = fopen(data_files{pos_idx}, 'rb');
        Data = fread(dataFile, 'float');
        fclose(dataFile);

        %% Initialize parameters
        frame_size = 2 * len; % Total samples per frame (I + Q)

        % Total number of frames
        total_frames = floor((length(Data) - offset) / (frame_size + offset));
        if total_frames < 1
            error('Data length is insufficient for even a single frame with the given parameters.');
        end

        %% Process frames
        doppler_frames = zeros(len, total_frames);
        for frame_idx = 1:total_frames
            start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
            frame_I = Data(start_idx:start_idx + len - 1);
            frame_Q = Data(start_idx + len:start_idx + frame_size - 1);
            frame_complex = frame_I + 1i * frame_Q; % Combine I and Q into complex frame
            doppler_frames(:, frame_idx) = frame_complex; %range FFT per frame information
        end
        %% apply fft to each column
        range_fft = fft(doppler_frames, [], 1); 
        
        % Check for NaN values in range_fft
        if any(isnan(range_fft(:)))
            warning('Data set %s contains NaN values. Skipping this data set.', labels{pos_idx});
            continue; % Skip this data set
        end

        %% Apply mti butterworth filter to range profile
        cutoff_freq = 0.02; 
        [b, a] = butter(9, cutoff_freq, 'high'); 
        range_fft_filtered = filtfilt(b, a, range_fft);

        %% Select range bins of interest (e.g., bins 5 to 20)
        range_bins_of_interest = 5:25;
        range_fft_selected = range_fft_filtered(range_bins_of_interest, :);

        %% Apply STFT to extract micro-Doppler signatures
        window_size = K; %fps/K = hamming window time
        overlap = round(ovlap * window_size); % ovlap = overlap percentage
        hamming_window = hamming(window_size);

        % Preallocate space for STFT output
        num_range_bins_selected = length(range_bins_of_interest);
        num_time_frames = floor((total_frames - window_size) / (window_size - overlap)) + 1;
        time_doppler_map = zeros(window_size, num_time_frames, num_range_bins_selected);

        % stft on range bins(REPLACE ASTERIXES WITH F,T IF NEEDED)
        for range_idx = 1:num_range_bins_selected
            time_series = range_fft_selected(range_idx, :);
            [S, ~,~] = stft(time_series, fps, 'Window', hamming_window, 'OverlapLength', overlap, 'FFTLength', window_size);
            time_doppler_map(:, :, range_idx) = abs(S);
        end

        % Sum across range bins to generate the final time-Doppler map
        time_doppler_map_sum = sum(time_doppler_map, 3);
        time_doppler_map_db = 20 * log10(rescale(time_doppler_map_sum));

        % Remove 112 rows from the top and bottom
        time_doppler_map_db_cut_rows = time_doppler_map_db(113:end-112, :); 
        %time_doppler_map_db_cut_rows = time_doppler_map_db;
        time_doppler_map_db_224x224 = imresize(time_doppler_map_db_cut_rows, [224, 224], 'bilinear');

        %% Save the 224x224 matrix as a CSV .txt file
        output_filename = sprintf('%s_Set%d_%s_MD224.txt', name, test_set, labels{pos_idx});
        writematrix(time_doppler_map_db_224x224, output_filename);
        disp(['Saved processed data for ', labels{pos_idx}, ' to ', output_filename]);
        doppler_resolution = fps / K;
        fprintf('Doppler resolution for %s: %.2f Hz\n', labels{pos_idx}, doppler_resolution);

        %% Store spectrogram data
        processed_spectrograms{pos_idx} = time_doppler_map_db_224x224;

        % Define the time axis (in seconds)
        %total_time = size(time_doppler_map_db_224x224, 2) / fps; % Total time in seconds
        time_axis = linspace(0, 224, size(time_doppler_map_db_224x224, 2)); % Time axis

        % Define the Doppler frequency axis (in Hz)
        doppler_axis = linspace(-112, 112, size(time_doppler_map_db_224x224, 1)); % Doppler frequency axis
        time_vectors{pos_idx} = time_axis; % Time axis from STFT
        frequency_vectors{pos_idx} = doppler_axis; % Frequency axis from STFT

        %% Plot the Time-Doppler Spectrogram in a subplot
        subplot(2, 3, pos_idx); % Arrange subplots in a 2x3 grid (5 positions)
        imagesc(time_axis,doppler_axis,time_doppler_map_db_224x224)
        %pcolor(time_axis, doppler_axis, time_doppler_map_db_224x224);
        %shading interp;
        axis xy; % Flip axes to keep the origin bottom-left
        colormap jet;
        colorbar;
        title(['Micro-Doppler Signature (' labels{pos_idx} ')']);
        xlabel('frame');
        ylabel('doppler bin');
        clim([-60, 0]); % Adjust color scale as needed
    end
end