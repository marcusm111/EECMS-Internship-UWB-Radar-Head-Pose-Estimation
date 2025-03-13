    function microdoppler224(sensor_id, data_files, K, ovlap, len, offset, fps, mindb, maxdb, run_id)
        % Process UWB radar data from five raw data files and save their micro-Doppler signatures.
        % Inputs:
        %   - sensor_id: sensor ID (e.g., 's1')
        %   - data_files: cell array of five radar data file paths
        %   - K: STFT window size
        %   - ovlap: STFT window overlapping factor
        %   - len: number of range bins
        %   - offset: radar data sample offset
        %   - fps: radar recording fps
        %   - run_id: test run ID (e.g., 'r1')
    
        labels = {'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8','p9'}; %p1-9: up, down, left 30, left 60, left 90, right 30, right 60, right 90, no movement
        num_positions = numel(data_files);
        %figure('WindowState', 'maximized');
        fh = figure();
        fh.WindowState = 'maximized';
    
        for pos_idx = 1:num_positions
            %% Load data
            dataFile = fopen(data_files{pos_idx}, 'rb');
            Data = fread(dataFile, 'float');
            fclose(dataFile);
    
            %% Initialize parameters and print frame count
            frame_size = 2 * len;
            total_frames = floor((length(Data) - offset) / (frame_size + offset));
            
            % Print frame count to console (NEW)
            [~, fname, ext] = fileparts(data_files{pos_idx});
            fprintf('Processed %s%s: %d frames\n', fname, ext, total_frames);
            
            if total_frames < 1
                error('Data length is insufficient for even a single frame with the given parameters.');
            end
    
            %% Process frames
            doppler_frames = zeros(len, total_frames);
            for frame_idx = 1:total_frames
                start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
                frame_I = Data(start_idx:start_idx + len - 1);
                frame_Q = Data(start_idx + len:start_idx + frame_size - 1);
                frame_complex = frame_I + 1i * frame_Q;
                doppler_frames(:, frame_idx) = frame_complex;
            end
    
            mti_doppler = zeros(len, total_frames); 
    
            %% MTI filter implementation for complex frames
            for frame_idx = 2:total_frames
                frame_complex_curr = doppler_frames(:, frame_idx);
                frame_complex_prev = doppler_frames(:, frame_idx-1);
                frame_complex_prev_rotated = -(frame_complex_prev);
                frame_filtered = frame_complex_curr + frame_complex_prev_rotated;
                mti_doppler(:, frame_idx - 1) = frame_filtered;
            end
            %% Apply FFT
            %range_fft = fft(doppler_frames, [], 1);
            %range_fft = fft(mti_doppler, [], 1); 
            %range_fft = doppler_frames;
            range_fft = mti_doppler;        
            %% High-pass filter
            cutoff_freq = 0.02;
            [b, a] = butter(4, cutoff_freq, 'high');
            range_fft_filtered = filtfilt(b, a, range_fft);
    
            %% Select range bins of interest
            range_bins_of_interest = 1:15;
            range_fft_selected = range_fft_filtered(range_bins_of_interest, :);
    
            %% Apply STFT
            window_size = K;
            overlap = round(ovlap * window_size);
            hamming_window = hamming(window_size);
            num_range_bins_selected = length(range_bins_of_interest);
            num_time_frames = floor((total_frames - window_size) / (window_size - overlap)) + 1;
            time_doppler_map = zeros(window_size, num_time_frames, num_range_bins_selected);
    
            for range_idx = 1:num_range_bins_selected
                time_series = range_fft_selected(range_idx, :);
                [S, ~, ~] = stft(time_series, fps, 'Window', hamming_window, 'OverlapLength', overlap, 'FFTLength', window_size);
                time_doppler_map(:, :, range_idx) = abs(S);
            end
    
            %% Sum across range bins
    %% Sum across range bins
    time_doppler_map_sum = sum(time_doppler_map, 3);
    
    %time_doppler_map_sum = rescale(time_doppler_map_sum,"InputMin",0,"InputMax",2.5);
    %^^ if ffting mti_doppler
    time_doppler_map_sum = rescale(time_doppler_map_sum,"InputMin",0,"InputMax",1.1);
    %^^ if not ffting mti_doppler
    time_doppler_map_db = 20 * log10(time_doppler_map_sum);
    time_doppler_map_db = max(time_doppler_map_db, mindb);
    time_doppler_map_db = min(time_doppler_map_db, maxdb);
            original_rows = size(time_doppler_map_db, 1);
            % Only crop rows if we have more than 224
            if original_rows > 224
                % Calculate symmetric crop to get central 224 rows
                crop_amount = original_rows - 224;
                start_row = floor(crop_amount/2) + 1;
                end_row = original_rows - floor(crop_amount/2);
                cropped_data = time_doppler_map_db(start_row:end_row, :);
            else
                % Use all rows if <= 224
                cropped_data = time_doppler_map_db;
            end
    
            % Final resize to 224x224
            time_doppler_map_db_224x224 = imresize(cropped_data, [224, 224], 'bilinear');
            % Round to 2 decimal places for ML efficiency
            time_doppler_map_db_224x224 = round(time_doppler_map_db_224x224 * 100) / 100;
    
            %% Save processed spectrogram for each position
            output_filename = sprintf('%s%sr%d.csv', sensor_id, labels{pos_idx}, run_id);
            writematrix(time_doppler_map_db_224x224, output_filename);
            disp(['Saved processed data to ', output_filename]);
    
            %% Plot with improved title (MODIFIED)
            time_axis = linspace(0, 224, size(time_doppler_map_db_224x224, 2));
            doppler_axis = linspace(-112, 112, size(time_doppler_map_db_224x224, 1));
            subplot(2, 5, pos_idx);
            imagesc(time_axis, doppler_axis, time_doppler_map_db_224x224);
            axis xy;
            colormap jet;
            colorbar;
            title(sprintf('md224(%s%sr%d)', sensor_id, labels{pos_idx}, run_id)); % Modified title
            xlabel('Frame');
            ylabel('Doppler Bin');
            clim([mindb, maxdb]);
        end
    end