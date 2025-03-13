% XeThru Radar Data Streaming Script for Three Sensors with Synchronized Video Recording

% Add ModuleConnector path
add_ModuleConnector_path();
clc;
clear;

%% User Configurations:
inputs = {'COM6','COM5','COM4'}; % COM ports for the three sensors
data_type = ModuleConnector.DataRecorderInterface.DataType_FloatDataType;
downconversion = 1; % 0: RF data output, 1: baseband IQ data output
rec = 1; % Enable recording
fps = 60; % Frames per second
record_duration = 2; % Recording duration in seconds
pause_duration = 2; % Pause between recordings in seconds

% Create directories if they do not exist
output_dir = 'recordings';
output_dir_vid = 'video_recordings';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(output_dir_vid, 'dir'), mkdir(output_dir_vid); end

% Initialize ZED Camera
zed = webcam('ZED 2');
zed.Resolution = zed.AvailableResolutions{1};
snapshot(zed); % Warm-up capture

% Radar parameters
dac_min = 949;
dac_max = 1100;
iterations = 16;
tx_center_frequency = 3;
tx_power = 2;
pulses_per_step = 5;
frame_area_start = 0;
frame_area_stop = 3;

% File labels
file_labels = {'up', 'down', 'left30', 'left60', 'left90', 'right30', 'right60', 'right90', 'nomovement'};
num_labels = length(file_labels);

% Initialize ModuleConnector Library
Lib = ModuleConnector.Library;
mc1 = []; mc2 = []; mc3 = [];

try
    % Initialize sensors
    mc1 = ModuleConnector.ModuleConnector(inputs{1}, 0);
    xep1 = mc1.get_xep();
    
    mc2 = ModuleConnector.ModuleConnector(inputs{2}, 0);
    xep2 = mc2.get_xep();
    
    mc3 = ModuleConnector.ModuleConnector(inputs{3}, 0);
    xep3 = mc3.get_xep();

    % Configure sensors
    sensors = {xep1, xep2, xep3};
    for i = 1:3
        xep = sensors{i};
        xep.x4driver_init();
        xep.x4driver_set_downconversion(downconversion);
        xep.x4driver_set_tx_center_frequency(tx_center_frequency);
        xep.x4driver_set_tx_power(tx_power);
        xep.x4driver_set_iterations(iterations);
        xep.x4driver_set_pulsesperstep(pulses_per_step);
        xep.x4driver_set_dac_min(dac_min);
        xep.x4driver_set_dac_max(dac_max);
        xep.x4driver_set_frame_area(frame_area_start, frame_area_stop);
        xep.x4driver_set_fps(fps);
    end

    %% Start Recording Loop
    for loop_count = 1:num_labels
        file_label = file_labels{loop_count};
        disp(['Starting recording cycle ', num2str(loop_count), ' (', file_label, ')']);

        % Start Video Recording
        vid_filename = fullfile(output_dir_vid, sprintf('%s.mp4', file_label));
        vid = VideoWriter(vid_filename, 'MPEG-4');
        vid.FrameRate = fps;
        open(vid);

        % Start Radar Recording
        recorder1 = mc1.get_data_recorder();
        recorder1.set_session_id(file_label);
        recorder1.start_recording(data_type, output_dir);

        recorder2 = mc2.get_data_recorder();
        recorder2.set_session_id(file_label);
        recorder2.start_recording(data_type, output_dir);

        recorder3 = mc3.get_data_recorder();
        recorder3.set_session_id(file_label);
        recorder3.start_recording(data_type, output_dir);

        start_time = tic;
        while toc(start_time) < record_duration
            img = snapshot(zed);
            writeVideo(vid, img);
            drawnow;
        end
        close(vid);

        % Stop Radar Recording
        recorder1.stop_recording(data_type);
        recorder2.stop_recording(data_type);
        recorder3.stop_recording(data_type);
        disp(['Stopped recording for cycle ', num2str(loop_count), ' (', file_label, ')']);

        % Pause before next recording
        pause(pause_duration);
    end

    % Stop data streaming
    for i = 1:3
        sensors{i}.x4driver_set_fps(0);
    end
    disp('Stopped data streaming.');

catch ME
    disp('Error occurred:');
    disp(ME.message);
end

%% Cleanup and Unload Library
try
    % Close ZED camera
    clear zed;

    % Clear and delete ModuleConnector objects
    delete(mc1); delete(mc2); delete(mc3);
    clear mc1 mc2 mc3 xep1 xep2 xep3 recorder1 recorder2 recorder3;

    % Unload the library if it was loaded
    if exist('Lib', 'var') && isvalid(Lib)
        Lib.unloadlib;
    end
    disp('Cleanup complete.');

catch cleanupME
    disp('Error during cleanup:');
    disp(cleanupME.message);
end
