
add_ModuleConnector_path();
clc;
clear;
close all;

%% User Configurations:
inputs = {'COM3','COM7','COM6'}; % COM ports for the two sensors
%up sensor: COM3
data_type = ModuleConnector.DataRecorderInterface.DataType_FloatDataType;
downconversion = 1; % 0: RF data output, 1: baseband IQ data output
rec = 1; % Enable recording
%NOTE FROM 26/02/25 WEDNESDAY: try increasing fps to 120 and decreasing
%window size to 30 to improve time resolution
fps = 60; % Frames per second
K = 56; %stft window size(number of doppler frequency bins)
start_run_num = 0;
maxdb = -5;
mindb = -35;
mindbF = -35;  
cam_fps = 6;
output_dir = 'recordings'; % Directory to save recordings
output_dir_vid = 'video recordings';
record_duration = 5; % Duration of recording in seconds
pause_duration = 2; % Pause between recordings in seconds
zed = webcam('ZED 2');
% Set video resolution
zed.Resolution = zed.AvailableResolutions{1};
% Get image size
[height width channels] = size(snapshot(zed));
% Warm up the webcam with a dummy snapshot
snapshot(zed); % Prevents initial delay in first capture
% Pre-initialize figure with placeholder image

dud_filename = fullfile(output_dir_vid, sprintf('dud.mp4')); 
dudvid = VideoWriter(dud_filename,'MPEG-4'); 
dudvid.FrameRate = cam_fps;
open(dudvid);

start_time = tic;
while toc(start_time) < record_duration
    img = snapshot(zed); 
    trueImg = img(:, width/2 +1: width, :); %for using right lens
    %trueImg = img(:, 1 : width/2, :); % for using left lens
    writeVideo(dudvid, trueImg);
    %imshow(trueImg); % Update existing image handle
    %drawnow('limitrate'); % Faster rendering
end
close(dudvid);

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

% Modified File labels for 7 positions
%{
file_labels = {
    'up',       % p1
    'down',     % p2 
    'left30',   %p3
    'left60',   % p3
    'left90',   % p4
    'right30',  %p6
    'right60',  % p7
    'right90',  % p8
    'nomovement'% p9
};
%}

file_labels = {
    'up',      % p1
    'down',
    'left90',
    'roll left',
    'right90',
    'roll right',
    'nomovement'
    };

num_labels = length(file_labels); 

% Ping sound configuration
fs = 44100; % Sampling frequency (Hz)
duration = 0.2; % Duration of the ping sound (seconds)
freq_start = 500; % Start frequency of the ping sound (Hz)
freq_stop = 1000; % Stop frequency of the ping sound (Hz)

% Generate ping sound
t = linspace(0, duration, fs * duration);
ping_sound = sin(2 * pi * linspace(freq_start, freq_stop, length(t)) .* t);
ping_sound1 = sin(2 * pi * 0.6 * linspace(freq_start, freq_stop, length(t)) .* t);

% Initialize counters for each label
label_counters = zeros(1, num_labels);

%% Initialize Radar Modules
Lib = ModuleConnector.Library;

try
    % Initialize connections for both sensors
    disp('Initializing Sensor 1 (RIGHT MOVEMENTS)...');
    mc1 = ModuleConnector.ModuleConnector(inputs{1}, 0);
    xep1 = mc1.get_xep();

    disp('Initializing Sensor 2 (MIDDLE)...');
    mc2 = ModuleConnector.ModuleConnector(inputs{2}, 0);
    xep2 = mc2.get_xep();
    
    disp('Initializing Sensor 3 (LEFT MOVEMENTS)...');
    mc3 = ModuleConnector.ModuleConnector(inputs{3}, 0);
    xep3 = mc3.get_xep();

    % Clear any existing data in the message queues
    while xep1.peek_message_data_float > 0
        xep1.read_message_data_float();
    end
    while xep2.peek_message_data_float > 0
        xep2.read_message_data_float();
    end
    while xep3.peek_message_data_float > 0
        xep3.read_message_data_float();
    end

    % Configure both sensors
    sensors = {xep1, xep2, xep3};
    for i = 1:3
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
    xep3.x4driver_set_fps(fps);

    %% Start Looped Recording
    
    fh = figure(1);
    clf(1);
    range_vector = [];
    loop_count = 0;
    recorded_files1 = cell(1,num_labels);
    recorded_files2 = cell(1,num_labels);
    recorded_files3 = cell(1,num_labels);

    % Recording loop
    while ishandle(fh)
        % Start recording for both sensors
        if rec
            loop_count = loop_count + 1;

            % Determine the label based on the loop count
            label_idx = mod(loop_count - 1, num_labels) + 1;
            file_label = file_labels{label_idx};
            label_counters(label_idx) = label_counters(label_idx) + 1;
            label_count = label_counters(label_idx) + start_run_num;

            disp(['Starting recording cycle ', num2str(loop_count), ' (', file_label, ')']);

            % Play ping sound for recording start
            sound(ping_sound, fs);

            % vid recording
            vid_filename = fullfile(output_dir_vid, sprintf('c1p%dr%d.mp4', label_idx, label_count)); 
            vid = VideoWriter(vid_filename,'MPEG-4'); 
            vid.FrameRate = cam_fps;
            open(vid);

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

            % Sensor 3 Recorder
            recorder3 = mc3.get_data_recorder();
            session_id3 = sprintf('s3p%dr%d', label_idx, label_count);
            recorder3.set_session_id(session_id3);
            recorder3.start_recording(data_type, output_dir);
        end

        start_time = tic;
        while toc(start_time) < record_duration && ishandle(fh)
            img = snapshot(zed); 
            trueImg = img(:, width/2 +1: width, :); %for using right lens
            %trueImg = img(:, 1 : width/2, :); % for using left lens
            writeVideo(vid, trueImg);
            %imshow(trueImg); % Update existing image handle
            %drawnow('limitrate'); % Faster rendering
        end
        close(vid);
        % Stop recording
        if rec
            recorder1.stop_recording(data_type);
            recorder2.stop_recording(data_type);
            recorder3.stop_recording(data_type);
            disp(['Stopped recording for cycle ', num2str(loop_count), ' (', file_label, ')']);
            % Play ping sound for recording stop
            sound(ping_sound1, fs);

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

            %SENSOR3 processing
            recording_folders3 = dir(fullfile(output_dir, 'xethru_recording_*s3*'));
            if ~isempty(recording_folders3)
                % Get the most recently modified folder
                [~, idx] = max([recording_folders3.datenum]);
                latest_folder3 = fullfile(output_dir, recording_folders3(idx).name);
        
                % Find and rename the recorded .dat file inside the folder
                dat_files3 = dir(fullfile(latest_folder3, 'xethru_datafloat_*.dat'));
                if ~isempty(dat_files3)
                    old_name3 = fullfile(latest_folder3, dat_files3(1).name);
                    new_name3 = fullfile(output_dir, sprintf('s3p%dr%d.dat', label_idx, label_count));
                    
                    % Move the file out of the folder
                    movefile(old_name3, new_name3);
                    disp(['Moved and renamed ', old_name3, ' to ', new_name3]);
    
                    % Store recorded file
                    recorded_files3{label_idx} = new_name3;
                end
        
                % Delete the metadata file if it exists
                meta_file3 = fullfile(latest_folder3, 'xethru_recording_meta.dat');
                if exist(meta_file3, 'file')
                    delete(meta_file3);
                    disp('Deleted metadata file.');
                end
        
                % Delete the now-empty folder
                rmdir(latest_folder3, 's');
                disp(['Deleted recording folder: ', latest_folder3]);
            else
                disp('No recording folder found.');
            end
        end

        % Process data when all five positions are recorded
        if mod(loop_count, num_labels) == 0
            %{
            sT_p1_file = fullfile(output_dir, sprintf('s1p1r%d.dat', label_count));
            sF_p1_file = fullfile(output_dir, sprintf('s2p1r%d.dat', label_count));
            sR_p1_file = fullfile(output_dir, sprintf('s3p1r%d.dat', label_count));
            cam_p1_file = fullfile(output_dir_vid, sprintf('c1p1r%d.mp4',label_count));

            % Check if the files exist and delete them
            if exist(cam_p1_file, 'file')
                delete(cam_p1_file);
                disp(['Deleted: ', cam_p1_file]);
            end
    
            % Check if the files exist and delete them
            if exist(sT_p1_file, 'file')
                delete(sT_p1_file);
                disp(['Deleted: ', sT_p1_file]);
            end
            if exist(sF_p1_file, 'file')
                delete(sF_p1_file);
                disp(['Deleted: ', sF_p1_file]);
            end
            if exist(sR_p1_file, 'file')
                delete(sR_p1_file);
                disp(['Deleted: ', sR_p1_file]);
            end
            %}
            disp('Processing recorded files for s1');
            disp(['run_id before function call: ', num2str(label_count)]);
            
            figure(2); % Use a separate figure for micro-Doppler images
            close(2);
            %microdoppler224('sR', recorded_files1, K, 0.95, 60, 3, fps, mindb,maxdb, label_count);
            
            asplit224('sR', recorded_files1, K, 0.95, 60, 3, fps,mindbF,maxdb,label_count,cam_fps,record_duration);
            disp('Processing recorded files for s2');
            disp(['run_id before function call: ', num2str(label_count)]);
            
            figure(3); % Use a separate figure for micro-Doppler images
            close(3);
            %microdoppler224('sF', recorded_files2, K, 0.95, 60, 3, fps,mindb,maxdb,label_count);
            
            asplit224('sF', recorded_files2, K, 0.95, 60, 3, fps,mindbF,maxdb,label_count,cam_fps,record_duration);
            disp('Processing complete. 224x224 micro-Doppler spectrum saved.');

            disp('Processing recorded files for s3');
            disp(['run_id before function call: ', num2str(label_count)]);
            
            figure(4); % Use a separate figure for micro-Doppler images
            close(4);
            %microdoppler224('sT', recorded_files3, K, 0.95, 60, 3, fps,mindbF,maxdb,label_count);
            
            asplit224('sT', recorded_files3, K, 0.95, 60, 3, fps,mindbF,maxdb,label_count,cam_fps,record_duration);
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
    xep3.x4driver_set_fps(0);
    disp('Stopped data streaming.');
catch ME
    % Handle errors
    disp('Error occurred:');
    disp(ME.message);
end

% Cleanup library and connections
clear mc1 mc2 mc3 xep1 xep2 xep3 recorder1 recorder2 recorder3 cam;
Lib.unloadlib;
disp('Cleanup complete.');
