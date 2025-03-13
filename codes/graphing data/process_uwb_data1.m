function process_uwb_data1(data_file, len, fps)
    % Process UWB radar data from a raw data file and output a spectrogram
    % Inputs:
    %   - data_file: Path to the .dat or .mat file containing the raw UWB data
    %   - len: Number of in-phase or quadrature samples per frame
    %   - fps: Recording frames per second of the radar data

    %% Load data
    dataFile = fopen(data_file,'rb');
    data = fread(dataFile,'float');
    fclose(dataFile);
    %% Initialize parameters
    frame_size = 2 * len; % Total samples per frame (I + Q)
    offset = 3; % Sample offset shift per frame

    % Total number of frames
    total_frames = floor((length(data) - offset) / (frame_size + offset));
    if total_frames < 1
        error('Data length is insufficient for even a single frame with the given parameters.');
    end

    %% Process frames
    processed_frames = zeros(len, total_frames); % Preallocate space for processed data

    for frame_idx = 1:total_frames
        start_idx = offset + (frame_idx - 1) * (frame_size + offset) + 1;
        end_idx = start_idx + frame_size - 1;

        if end_idx > length(data)
            warning('Skipping incomplete frame at the end of the dataset.');
            break;
        end

        % Extract in-phase (I) and quadrature (Q) data
        frame_I = data(start_idx:start_idx + len - 1);
        frame_Q = data(start_idx + len:start_idx + frame_size - 1);

        % Reconstruct complex radar frame
        frame_complex = frame_I + 1i * frame_Q;

        % Store the magnitude of the frame
        processed_frames(:, frame_idx) = abs(frame_complex);
    end

    %% Visualization
    % Convert sample index to range (in meters)
    range_per_sample = 3.0092 / len; % Range corresponding to each sample
    range = (0:len-1) * range_per_sample; % Range vector

    % Generate the spectrogram
    figure;
    imagesc(1:total_frames, range, processed_frames);
    axis xy; % Ensure the origin is at the bottom-left
    colormap jet;
    colorbar;
    title('Radar Data Spectrogram');
    xlabel('Frame Number');
    ylabel('Range (m)');
    clim([min(processed_frames(:)), max(processed_frames(:))]);

    %% Save processed frames
    %save('processed_uwb_frames.mat', 'processed_frames');
    %disp('Processed frames saved to processed_uwb_frames.mat');
end
