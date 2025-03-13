% XeThru Radar Data Streaming Script with Loop for Recording
% This script reads and processes radar data and ensures proper initialization and cleanup to run multiple times consecutively.

% Add ModuleConnector path
add_ModuleConnector_path();
clc;
clear;

%% User configurations:
input = 'COM7'; % COM port of the XeThru module
data_type = ModuleConnector.DataRecorderInterface.DataType_FloatDataType;
downconversion = 1; % 0: RF data output, 1: baseband IQ data output
rec = 1; % Enable recording
fps = 60; % Frames per second
output_dir = 'recordings'; % Directory to save recordings
record_duration = 3; % Duration of recording in seconds
pause_duration = 2; % Duration to pause between recordings in seconds

% Radar parameters
dac_min = 949;
dac_max = 1100;
iterations = 16;
tx_center_frequency = 3;
tx_power = 2;
pulses_per_step = 5;
frame_area_offset = 0.18;
frame_area_start = 0;
frame_area_stop = 3;

% File labels for naming saved recordings
file_labels = {'p1', 'p2', 'p3', 'p4', 'p5'};
num_labels = length(file_labels);
label_counters = zeros(1, num_labels);

% Ping sound configuration
fs = 44100; % Sampling frequency (Hz)
duration = 0.2; % Duration of the ping sound (seconds)
freq_start = 500; % Start frequency of the ping sound (Hz)
freq_stop = 1000; % Stop frequency of the ping sound (Hz)

% Generate ping sound
t = linspace(0, duration, fs * duration);
ping_sound = sin(2 * pi * linspace(freq_start, freq_stop, length(t)) .* t);

%% Initialize the radar module
Lib = ModuleConnector.Library;

try
    % Create ModuleConnector object
    mc = ModuleConnector.ModuleConnector(input, 0);

    % Get XEP interface
    xep = mc.get_xep();

    % Clear any existing data in the message queue
    while xep.peek_message_data_float > 0
        xep.read_message_data_float();
    end

    % Stop any previous sensor mode
    try
        app = mc.get_x4m300();
        app.set_sensor_mode('stop'); % Stop the sensor
        app.set_sensor_mode('XEP'); % Set manual mode
    catch
        disp('Sensor already in XEP mode or failed to stop previous mode.');
    end

    % Initialize the radar chip
    xep.x4driver_init();

    % Set radar parameters
    xep.x4driver_set_downconversion(downconversion);
    xep.x4driver_set_tx_center_frequency(tx_center_frequency);
    xep.x4driver_set_tx_power(tx_power);
    xep.x4driver_set_iterations(iterations);
    xep.x4driver_set_pulsesperstep(pulses_per_step);
    xep.x4driver_set_dac_min(dac_min);
    xep.x4driver_set_dac_max(dac_max);
    xep.x4driver_set_frame_area_offset(frame_area_offset);
    xep.x4driver_set_frame_area(frame_area_start, frame_area_stop);

    % Set FPS to start streaming
    xep.x4driver_set_fps(fps);

    %% Start looped recording
    fh = figure(1);
    clf(1);
    ph_rf = plot(NaN, NaN);
    title('Radar Raw Data');
    xlabel('Range [m]');
    if downconversion == 1
        ylim([0 0.08]);
    else
        ylim([-0.08 0.08]);
    end

    range_vector = [];
    loop_count = 0;
    recorded_files = cell(1, num_labels);

    % Recording loop
    while ishandle(fh)
        % Start recording
        if rec
            label_idx = mod(loop_count, num_labels) + 1;
            file_label = file_labels{label_idx};
            label_counters(label_idx) = label_counters(label_idx) + 1;
            label_count = label_counters(label_idx);
            loop_count = loop_count + 1;

            disp(['Starting recording cycle ', num2str(loop_count)]);
            sound(ping_sound, fs);
            session_id = sprintf('s1p%dr%d', label_idx, label_count);
            recorder = mc.get_data_recorder();
            recorder.set_session_id(session_id);
            recorder.start_recording(data_type, output_dir);
        end

        % Record for the specified duration
        start_time = tic;
        while toc(start_time) < record_duration && ishandle(fh)
            [int, len, frame_count, data] = xep.read_message_data_float();
            if downconversion == 1
                % Process IQ data
                len = len / 2;
                i_vec = data(1:len);
                q_vec = data(len+1:end);
                iq_vec = i_vec + 1i * q_vec;
                data = abs(iq_vec);
            end
            if isempty(range_vector)
                % Generate range vector on the first run
                range_vector = linspace(frame_area_start, frame_area_stop, len);
                ph_rf.XData = range_vector;
            end
            ph_rf.YData = data;
            title(['Frame Count: ', num2str(frame_count)]);
            drawnow;
        end

        % Stop recording
        if rec
            recorder.stop_recording(data_type);
            disp(['Stopped recording for cycle ', num2str(loop_count)]);
        
            sound(ping_sound, fs);
            % Find the latest recording folder
            recording_folders = dir(fullfile(output_dir, 'xethru_recording_*'));
            if ~isempty(recording_folders)
                % Get the most recently modified folder
                [~, idx] = max([recording_folders.datenum]);
                latest_folder = fullfile(output_dir, recording_folders(idx).name);
        
                % Find and rename the recorded .dat file inside the folder
                dat_files = dir(fullfile(latest_folder, 'xethru_datafloat_*.dat'));
                if ~isempty(dat_files)
                    old_name = fullfile(latest_folder, dat_files(1).name);
                    new_name = fullfile(output_dir, sprintf('s1p%dr%d.dat', label_idx, label_count));
                    
                    % Move the file out of the folder
                    movefile(old_name, new_name);
                    disp(['Moved and renamed ', old_name, ' to ', new_name]);

                    % Store recorded file
                    recorded_files{label_idx} = new_name;
                end
        
                % Delete the metadata file if it exists
                meta_file = fullfile(latest_folder, 'xethru_recording_meta.dat');
                if exist(meta_file, 'file')
                    delete(meta_file);
                    disp('Deleted metadata file.');
                end
        
                % Delete the now-empty folder
                rmdir(latest_folder, 's');
                disp(['Deleted recording folder: ', latest_folder]);
            else
                disp('No recording folder found.');
            end
        end

        
        % Process data when all five positions are recorded
        if mod(loop_count, num_labels) == 0
            disp('Processing recorded files...');
            disp(['run_id before function call: ', num2str(label_count)]); 
            figure(2); % Use a separate figure for micro-Doppler images
            close(2);
            microdoppler224('s1', recorded_files,56,0.95,60,3,60,-35,0,1);
            disp('Processing complete. 224x224 micro-Doppler spectrum saved.');
        end

        % Pause for the specified duration
        disp(['Pausing for ', num2str(pause_duration), ' seconds...']);
        pause(pause_duration);
    end

    %% Cleanup after running
    xep.x4driver_set_fps(0); % Stop streaming
    app.set_sensor_mode('stop'); % Ensure the module is stopped
catch ME
    % Handle errors
    disp('Error occurred:');
    disp(ME.message);
end

% Cleanup library and connections
clear mc xep recorder;
Lib.unloadlib;
disp('Cleanup complete.');
