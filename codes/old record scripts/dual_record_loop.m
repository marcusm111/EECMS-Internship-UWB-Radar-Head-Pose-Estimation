% XeThru Radar Data Streaming Script for Two Sensors with Custom File Naming and Ping Sounds
% This script reads and processes radar data from two sensors, ensures proper file naming,
% and plays a ping sound each time recording starts and stops.

% Add ModuleConnector path
add_ModuleConnector_path();
clc;
clear;

%% User Configurations:
inputs = {'COM6', 'COM7'}; % COM ports for the two sensors
data_type = ModuleConnector.DataRecorderInterface.DataType_FloatDataType;
downconversion = 1; % 0: RF data output, 1: baseband IQ data output
rec = 1; % Enable recording
fps = 500; % Frames per second
output_dir = 'recordings'; % Directory to save recordings
record_duration = 3; % Duration of recording in seconds
pause_duration = 2; % Pause between recordings in seconds

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
file_labels = {'up', 'down', 'left', 'right', 'nomovement'};
num_labels = length(file_labels);

% Ping sound configuration
fs = 44100; % Sampling frequency (Hz)
duration = 0.2; % Duration of the ping sound (seconds)
freq_start = 500; % Start frequency of the ping sound (Hz)
freq_stop = 1000; % Stop frequency of the ping sound (Hz)

% Generate ping sound
t = linspace(0, duration, fs * duration);
ping_sound = sin(2 * pi * linspace(freq_start, freq_stop, length(t)) .* t);

% Initialize counters for each label
label_counters = zeros(1, num_labels);

%% Initialize Radar Modules
Lib = ModuleConnector.Library;

try
    % Initialize connections for both sensors
    disp('Initializing Sensor 1 (left)...');
    mc1 = ModuleConnector.ModuleConnector(inputs{1}, 0);
    xep1 = mc1.get_xep();

    disp('Initializing Sensor 2 (right)...');
    mc2 = ModuleConnector.ModuleConnector(inputs{2}, 0);
    xep2 = mc2.get_xep();

    % Clear any existing data in the message queues
    while xep1.peek_message_data_float > 0
        xep1.read_message_data_float();
    end
    while xep2.peek_message_data_float > 0
        xep2.read_message_data_float();
    end

    % Configure both sensors
    sensors = {xep1, xep2};
    for i = 1:2
        xep = sensors{i};

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
    end

    % Set FPS to start streaming for both sensors
    xep1.x4driver_set_fps(fps);
    xep2.x4driver_set_fps(fps);

    %% Start Looped Recording
    fh = figure(1);
    clf(1);
    subplot(2, 1, 1);
    ph_rf1 = plot(NaN, NaN);
    title('Sensor 1 - Radar Raw Data');
    xlabel('Range [m]');
    if downconversion == 1
        ylim([0 0.08]);
    else
        ylim([-0.08 0.08]);
    end

    subplot(2, 1, 2);
    ph_rf2 = plot(NaN, NaN);
    title('Sensor 2 - Radar Raw Data');
    xlabel('Range [m]');
    if downconversion == 1
        ylim([0 0.08]);
    else
        ylim([-0.08 0.08]);
    end

    range_vector = [];
    loop_count = 0;
    recorded_files1 = cell(1,num_labels);
    recorded_files2 = cell(1,num_labels);

    % Recording loop
    while ishandle(fh)
        % Start recording for both sensors
        if rec
            loop_count = loop_count + 1;

            % Determine the label based on the loop count
            label_idx = mod(loop_count - 1, num_labels) + 1;
            file_label = file_labels{label_idx};
            label_counters(label_idx) = label_counters(label_idx) + 1;
            label_count = label_counters(label_idx);

            disp(['Starting recording cycle ', num2str(loop_count), ' (', file_label, ')']);

            % Play ping sound for recording start
            sound(ping_sound, fs);

            % Sensor 1 Recorder
            recorder1 = mc1.get_data_recorder();
            session_id1 = sprintf('s1p%dr%d', label_idx, label_count);
            recorder1.set_session_id(session_id1);
            recorder1.start_recording(data_type, output_dir);

            % Sensor 2 Recorder
            recorder2 = mc2.get_data_recorder();
            session_id2 = sprintf('s2p%dr%d', label_idx, label_count);
            recorder2.set_session_id(session_id2);
            recorder2.start_recording(data_type, output_dir);
        end

        % Record for the specified duration
        start_time = tic;
        while toc(start_time) < record_duration && ishandle(fh)
            % Sensor 1 Data
            [~, len1, ~, data1] = xep1.read_message_data_float();
            if len1 > 0
                len1 = len1 / 2;
                i_vec1 = data1(1:len1);
                q_vec1 = data1(len1+1:end);
                iq_vec1 = i_vec1 + 1i * q_vec1;
                data1 = abs(iq_vec1);
                if isempty(range_vector)
                    range_vector = linspace(frame_area_start, frame_area_stop, len1);
                    ph_rf1.XData = range_vector;
                end
                ph_rf1.YData = data1;
            end

            % Sensor 2 Data
            [~, len2, ~, data2] = xep2.read_message_data_float();
            if len2 > 0
                len2 = len2 / 2;
                i_vec2 = data2(1:len2);
                q_vec2 = data2(len2+1:end);
                iq_vec2 = i_vec2 + 1i * q_vec2;
                data2 = abs(iq_vec2);
                ph_rf2.YData = data2;
                ph_rf2.XData = linspace(frame_area_start, frame_area_stop, len2);
            end

            drawnow;
        end

        % Stop recording
        if rec
            recorder1.stop_recording(data_type);
            recorder2.stop_recording(data_type);
            disp(['Stopped recording for cycle ', num2str(loop_count), ' (', file_label, ')']);
            
            % Play ping sound for recording stop
            sound(ping_sound, fs);

            %SENSOR1 .dat processing
            recording_folders1 = dir(fullfile(output_dir, 'xethru_recording_*s1*'));
            if ~isempty(recording_folders1)
                % Get the most recently modified folder
                [~, idx] = max([recording_folders1.datenum]);
                latest_folder1 = fullfile(output_dir, recording_folders1(idx).name);
        
                % Find and rename the recorded .dat file inside the folder
                dat_files1 = dir(fullfile(latest_folder1, 'xethru_datafloat_*.dat'));
                if ~isempty(dat_files1)
                    old_name1 = fullfile(latest_folder1, dat_files1(1).name);
                    new_name1 = fullfile(output_dir, sprintf('s1p%dr%d.dat', label_idx, label_count));
                    
                    % Move the file out of the folder
                    movefile(old_name1, new_name1);
                    disp(['Moved and renamed ', old_name1, ' to ', new_name1]);
    
                    % Store recorded file
                    recorded_files1{label_idx} = new_name1;
                end
        
                % Delete the metadata file if it exists
                meta_file1 = fullfile(latest_folder1, 'xethru_recording_meta.dat');
                if exist(meta_file1, 'file')
                    delete(meta_file1);
                    disp('Deleted metadata file.');
                end
        
                % Delete the now-empty folder
                rmdir(latest_folder1, 's');
                disp(['Deleted recording folder: ', latest_folder1]);
            else
                disp('No recording folder found.');
            end

            %SENSOR2 processing
            recording_folders2 = dir(fullfile(output_dir, 'xethru_recording_*s2*'));
            if ~isempty(recording_folders2)
                % Get the most recently modified folder
                [~, idx] = max([recording_folders2.datenum]);
                latest_folder2 = fullfile(output_dir, recording_folders2(idx).name);
        
                % Find and rename the recorded .dat file inside the folder
                dat_files2 = dir(fullfile(latest_folder2, 'xethru_datafloat_*.dat'));
                if ~isempty(dat_files2)
                    old_name2 = fullfile(latest_folder2, dat_files2(1).name);
                    new_name2 = fullfile(output_dir, sprintf('s2p%dr%d.dat', label_idx, label_count));
                    
                    % Move the file out of the folder
                    movefile(old_name2, new_name2);
                    disp(['Moved and renamed ', old_name2, ' to ', new_name2]);
    
                    % Store recorded file
                    recorded_files2{label_idx} = new_name2;
                end
        
                % Delete the metadata file if it exists
                meta_file2 = fullfile(latest_folder2, 'xethru_recording_meta.dat');
                if exist(meta_file2, 'file')
                    delete(meta_file2);
                    disp('Deleted metadata file.');
                end
        
                % Delete the now-empty folder
                rmdir(latest_folder2, 's');
                disp(['Deleted recording folder: ', latest_folder2]);
            else
                disp('No recording folder found.');
            end
        end

        % Process data when all five positions are recorded
        if mod(loop_count, num_labels) == 0
            disp('Processing recorded files for s1');
            disp(['run_id before function call: ', num2str(label_count)]);
            figure(2); % Use a separate figure for micro-Doppler images
            close(2);
            microdoppler224('s1', recorded_files1, 448, 0.95, 60, 3, 500, label_count);
            
            disp('Processing recorded files for s2');
            disp(['run_id before function call: ', num2str(label_count)]);
            figure(3); % Use a separate figure for micro-Doppler images
            close(3);
            microdoppler224('s2', recorded_files2, 448, 0.95, 60, 3, 500, label_count);
            disp('Processing complete. 224x224 micro-Doppler spectrum saved.');
        end
            
        % Pause between recordings
        disp(['Pausing for ', num2str(pause_duration), ' seconds...']);
        pause(pause_duration)
        %{
        pause_start = tic;
        while toc(pause_start) < pause_duration
            drawnow; % Allows MATLAB to process events
        end
        %}
    end

    %% Cleanup after running
    xep1.x4driver_set_fps(0); % Stop streaming for Sensor 1
    xep2.x4driver_set_fps(0); % Stop streaming for Sensor 2
    disp('Stopped data streaming.');
catch ME
    % Handle errors
    disp('Error occurred:');
    disp(ME.message);
end

% Cleanup library and connections
clear mc1 mc2 xep1 xep2 recorder1 recorder2;
Lib.unloadlib;
disp('Cleanup complete.');
