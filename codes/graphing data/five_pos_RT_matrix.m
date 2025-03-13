function five_pos_RT_matrix(test_set, data_up, data_down, data_left, data_right, data_nomovement, len, set_range, range_lim, offset, fps)
    % Process UWB radar data from five raw data files and visualize their range-time spectrograms
    % Inputs:
    %   - test_set: data set number
    %   - data_up, data_down, data_left, data_right, data_nomovement: Paths to the radar data files
    %   - len: Number of in-phase or quadrature samples per frame/number of range bins
    %   - set_range: The operating range of the UWB sensor set when recording data
    %   - range_lim: The range limit to be specified for plots
    %   - offset: The sample index offset for each radar frame
    %   - fps: Frames per second recording speed of radar

    % Store file paths and labels for iteration
    data_files = {data_up, data_down, data_left, data_right, data_nomovement};
    labels = {'Up', 'Down', 'Left90', 'Right90', 'No Movement'};
    
    % Initialize parameters
    num_positions = numel(data_files);
    processed_spectrograms = cell(1, num_positions);
    time_vectors = cell(1, num_positions);

    for pos_idx = 1:num_positions
        %% Load data
        dataFile = fopen(data_files{pos_idx}, 'rb');
        Data = fread(dataFile, 'float');
        fclose(dataFile);

        frame_size = 2 * len; % Total samples per frame (I + Q)

        % Total number of frames
        total_frames = floor((length(Data) - offset) / (frame_size + offset));
        if total_frames < 1
            error('Data length is insufficient for even a single frame with the given parameters.');
        end

        %% Process frames
        processed_frames = zeros(len, total_frames); % Preallocate space for processed data

        for frame_idx = 1:total_frames
            start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
            end_idx = start_idx + frame_size - 1;

            if end_idx > length(Data)
                warning('Skipping incomplete frame at the end of the dataset.');
                break;
            end

            % Extract in-phase (I) and quadrature (Q) data
            frame_I = Data(start_idx:start_idx + len - 1);
            frame_Q = Data(start_idx + len:start_idx + frame_size - 1);

            % Reconstruct complex radar frame
            frame_complex = frame_I + 1i * frame_Q;

            % Store the magnitude of the frame
            processed_frames(:, frame_idx) = abs(frame_complex);
        end

        %% Apply MTI filtering
        mti_frames = zeros(len, total_frames - 1); % MTI will result in one less frame

        for frame_idx = 2:total_frames
            frame_complex_curr = processed_frames(:, frame_idx);
            frame_complex_prev = processed_frames(:, frame_idx - 1);

            % Rotate the previous frame by 180 degrees (negate complex conjugate)
            frame_complex_prev_rotated = -conj(frame_complex_prev);

            % Apply MTI filter: Add rotated previous frame to current frame
            frame_filtered = frame_complex_curr + frame_complex_prev_rotated;

            % Store the magnitude of the filtered frame
            mti_frames(:, frame_idx - 1) = abs(frame_filtered);
        end

        %% Apply low-pass Doppler filtering
        window_size = 5; 
        half_window = floor(window_size / 2);

        % Preallocate filtered frames
        low_pass_frames = zeros(size(mti_frames));

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

        % Rescale low-pass frames
        low_pass_frames = rescale(low_pass_frames);
    
        %% Save low-pass frames to a CSV file
        output_filename = sprintf('Set%d_%s_Data.txt', test_set, labels{pos_idx});
        writematrix(20*log10(low_pass_frames), output_filename)
        %csvwrite(output_filename, low_pass_frames); % Save as a CSV file
        disp(['Saved processed data for ', labels{pos_idx}, ' to ', output_filename]);
        %% Prepare spectrogram data
        range_per_sample = set_range / len; % Range corresponding to each sample
        range = (0:len-1) * range_per_sample; % Range vector
        total_time = total_frames / fps; % Total time in seconds
        time_vector = linspace(0, total_time, total_frames - 1);
        time_vectors{pos_idx} = time_vector;

        % Store spectrogram data
        processed_spectrograms{pos_idx} = low_pass_frames;
    end

    %% Visualization
    figure;
    for pos_idx = 1:num_positions
        subplot(3, 2, pos_idx); % Arrange subplots in a 3x2 grid
        spectrogram_data = 20 * log10(processed_spectrograms{pos_idx}); % Convert to dB
        time_vector = time_vectors{pos_idx};

        % Generate the spectrogram
        pcolor(time_vector,range,spectrogram_data);
        shading interp;
        axis xy; % Ensure the origin is at the bottom-left
        colormap jet;
        colorbar;
        title(['Radar Data Spectrogram (' labels{pos_idx} ')']);
        xlabel('Time (s)');
        ylabel('Range (m)');
        ylim([0 range_lim]); % Limit the y-axis to the specified range limit
        clim([-35 0]); % Adjust color limits for better visualization
    end
end