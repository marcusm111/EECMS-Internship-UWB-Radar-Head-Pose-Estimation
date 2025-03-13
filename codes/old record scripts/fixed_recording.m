% XeThru Radar Data Streaming Script
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
fps = 500; % Frames per second
output_dir = 'recordings'; % Directory to save recordings

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

    % Set up recording if enabled
    if rec
        mkdir(output_dir);
        recorder = mc.get_data_recorder();
        recorder.set_session_id('Float_Data_recording');
        recorder.start_recording(data_type, output_dir);
    end

    %% Visualize radar data
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
    while ishandle(fh)
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

    %% Cleanup after running
    if rec
        recorder.stop_recording(data_type);
    end
    xep.x4driver_set_fps(0); % Stop streaming
    app.set_sensor_mode('stop'); % Ensure the module is stopped
catch ME
    % Handle errors and cleanup
    disp('Error occurred:');
    disp(ME.message);
end

% Cleanup library and connections
if exist('mc', 'var')
    clear mc;
end
clear mc;
clear xep;
clear app;
clear recorder;
Lib.unloadlib;
clear Lib;