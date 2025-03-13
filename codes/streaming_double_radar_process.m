% XeThru Radar Data Streaming Script for Two Sensors with Real-Time Micro-Doppler Processing
% This script streams raw radar data from two sensors. In addition to real-time
% raw data visualization, it continuously processes Sensor 1 data to compute and update
% a micro-Doppler spectrum plot. During each recording cycle, each computed micro-Doppler
% spectrum frame is appended to an accumulator. At the end of the recording cycle, a color plot
% is generated with the x-axis as the micro-Doppler frame number, the y-axis as the Doppler frequency,
% and the color representing Doppler power (in dB).

add_ModuleConnector_path();
clc;
clear;

%% User Configurations:
inputs = {'COM6', 'COM7'};            % COM ports for the two sensors
data_type = ModuleConnector.DataRecorderInterface.DataType_FloatDataType;
downconversion = 1;                  % 0: RF data output, 1: baseband IQ data output
rec = 1;                             % Enable recording
fps = 30;                            % Frames per second for raw streaming (for visualization)
% (Note: For the micro-Doppler STFT, we also use fps.)
K = 28;                              % STFT window size (and number of frames used for micro-Doppler processing)
ovlap = 0.8;                         % Overlap factor for STFT
output_dir = 'recordings';           % Directory to save recordings
record_duration = 5;                % Duration of each recording cycle (seconds)
pause_duration = 2;                  % Pause between recording cycles (seconds)

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

% File labels for naming saved recordings (for the five head positions)
file_labels = {'up', 'down', 'left', 'right', 'nomovement'};
num_labels = length(file_labels);
label_counters = zeros(1, num_labels);

% Ping sound configuration
sound_fs = 44100;                 % Sound sampling frequency (Hz)
sound_duration = 0.2;             % Duration of the ping sound (seconds)
freq_start = 500;                 % Start frequency (Hz)
freq_stop = 1000;                 % Stop frequency (Hz)
t_sound = linspace(0, sound_duration, sound_fs * sound_duration);
ping_sound = sin(2 * pi * linspace(freq_start, freq_stop, length(t_sound)) .* t_sound);

%% Initialize Figures:
% Raw radar data figure
fh_raw = figure(1);
clf(fh_raw);
subplot(2,1,1);
ph_rf1 = plot(NaN, NaN);
title('Sensor 1 - Radar Raw Data');
xlabel('Range [m]');
if downconversion == 1
    ylim([0 0.08]);
else
    ylim([-0.08 0.08]);
end

subplot(2,1,2);
ph_rf2 = plot(NaN, NaN);
title('Sensor 2 - Radar Raw Data');
xlabel('Range [m]');
if downconversion == 1
    ylim([0 0.08]);
else
    ylim([-0.08 0.08]);
end

% Figure for real-time micro-Doppler (Sensor 1)
fh_dopp = figure(2);
clf(fh_dopp);
% We'll update this figure with a real-time plot (e.g. using plot)
% and accumulate computed micro-Doppler frames for a final color plot.
ph_dopp = [];  % For real-time update (line plot)
% Initialize accumulator for micro-Doppler frames (each column is one computed spectrum)
md_accum = []; 

%% Initialize a buffer for Sensor 1 raw complex frames (for micro-Doppler processing)
raw_buffer = [];    % Each column will be one frame (complex IQ data)
window_frames = K;  % Use K frames as the processing window

%% Initialize Radar Module
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
        xep.x4driver_init();
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

    %% Start Looped Recording and Real-Time Micro-Doppler Processing
    loop_count = 0;
    range_vector = []; % will be set from the first received frame

    % Recording loop: for each cycle, record data for record_duration seconds,
    % update raw plots, and update the micro-Doppler plot.
    while ishandle(fh_raw) && ishandle(fh_dopp)
        % Start recording for both sensors
        if rec
            loop_count = loop_count + 1;
            % Determine the label based on loop_count (cyclic over file_labels)
            label_idx = mod(loop_count - 1, num_labels) + 1;
            file_label = file_labels{label_idx};
            label_counters(label_idx) = label_counters(label_idx) + 1;
            label_count = label_counters(label_idx);

            disp(['Starting recording cycle ', num2str(loop_count), ' (', file_label, ')']);

            % Play ping sound for recording start
            sound(ping_sound, sound_fs);

            % Sensor 1 Recorder
            recorder1 = mc1.get_data_recorder();
            session_id1 = sprintf('s1%sr%d', file_label, label_count);
            recorder1.set_session_id(session_id1);
            recorder1.start_recording(data_type, output_dir);

            % Sensor 2 Recorder
            recorder2 = mc2.get_data_recorder();
            session_id2 = sprintf('s2%sr%d', file_label, label_count);
            recorder2.set_session_id(session_id2);
            recorder2.start_recording(data_type, output_dir);
        end

        % Clear the raw buffer for Sensor 1 and micro-Doppler accumulator for the new cycle
        raw_buffer = [];
        md_accum = [];

        % Set up a timer for the recording duration
        start_time = tic;
        while toc(start_time) < record_duration && ishandle(fh_raw) && ishandle(fh_dopp)
            % --- Sensor 1 Data Processing ---
            [~, len1, ~, data1] = xep1.read_message_data_float();
            if len1 > 0
                len1 = len1 / 2;
                i_vec1 = data1(1:len1);
                q_vec1 = data1(len1+1:end);
                iq_vec1 = i_vec1 + 1i * q_vec1;
                raw_frame = iq_vec1;  % raw complex frame for micro-Doppler processing
                data1_abs = abs(iq_vec1); % magnitude for raw plot

                % Update raw plot for Sensor 1
                if isempty(range_vector)
                    range_vector = linspace(frame_area_start, frame_area_stop, len1);
                    ph_rf1.XData = range_vector;
                end
                ph_rf1.YData = data1_abs;

                % Append the raw complex frame to the buffer
                raw_buffer = [raw_buffer, raw_frame]; %#ok<AGROW>
            end

            % --- Sensor 2 Data Processing (raw plot only) ---
            [~, len2, ~, data2] = xep2.read_message_data_float();
            if len2 > 0
                len2 = len2 / 2;
                i_vec2 = data2(1:len2);
                q_vec2 = data2(len2+1:end);
                iq_vec2 = i_vec2 + 1i * q_vec2;
                data2_abs = abs(iq_vec2);
                ph_rf2.YData = data2_abs;
                ph_rf2.XData = linspace(frame_area_start, frame_area_stop, len2);
            end

            drawnow;

            % --- Real-Time Micro-Doppler Processing for Sensor 1 ---
            if size(raw_buffer,2) >= window_frames
                % Use the last "window_frames" columns for processing
                window_data = raw_buffer(:, end-window_frames+1:end);

                % 1. Apply a simple MTI filter: difference between successive frames
                mti_doppler = zeros(size(window_data));
                for f = 2:window_frames
                    mti_doppler(:, f-1) = window_data(:, f) - window_data(:, f-1);
                end

                % 2. Optionally, apply a high-pass Butterworth filter to remove residual clutter
                cutoff_freq = 0.02;  % normalized cutoff
                [b, a] = butter(9, cutoff_freq, 'high');
                mti_doppler_filtered = filtfilt(b, a, mti_doppler);

                % 3. Select a representative time series from the filtered data.
                range_bins_of_interest = 1:15;
                selected_data = mean(mti_doppler_filtered(range_bins_of_interest, :), 1);

                % 4. Apply STFT to the selected time series to compute the Doppler spectrum.
                window_size_stft = K;
                overlap_val = round(ovlap * window_size_stft);
                hamming_window = hamming(window_size_stft);
                [S, F, T] = stft(selected_data, fps, 'Window', hamming_window, 'OverlapLength', overlap_val, 'FFTLength', window_size_stft);
                % Take the most recent time slice (last column)
                spectrum = abs(S(:, end));

                % 5. Update the real-time Doppler plot (line plot)
                if isempty(ph_dopp) || ~ishandle(ph_dopp)
                    fh_dopp = figure(3);
                    clf(fh_dopp);
                    ph_dopp = plot(F, spectrum, 'LineWidth', 2);
                    title('Real-Time Micro-Doppler Spectrum (Sensor 1)');
                    xlabel('Doppler Frequency (Hz)');
                    ylabel('Power (dB)');
                    grid on;
                    xlim([-15 15]);
                    ylim([0, 0.1]);
                else
                    set(ph_dopp, 'XData', F, 'YData', spectrum);
                    figure(fh_dopp);
                end
                drawnow;

                % 6. Append the computed spectrum to the micro-Doppler accumulator.
                % Each column in md_accum corresponds to one computed micro-Doppler frame.
                md_accum = [md_accum, spectrum]; %#ok<AGROW>
            end
        end

        % At the end of the recording cycle, generate a color plot of the accumulated micro-Doppler frames.
        if ~isempty(md_accum)
            fh_md_final = figure(4);
            clf(fh_md_final);
            imagesc(1:size(md_accum,2), F, md_accum);
            axis xy;
            colormap jet;
            colorbar;
            title('Accumulated Micro-Doppler Spectrum (Sensor 1)');
            xlabel('Micro-Doppler Frame Number');
            ylabel('Doppler Frequency (Hz)');
            clim([0 0.1]);
            drawnow;
            % Optionally, you could save the md_accum matrix to a file here.
            % For example: writematrix(md_accum, sprintf('md_accum_cycle%d.txt', loop_count));
        end

        % Stop recording for this cycle
        if rec
            recorder1.stop_recording(data_type);
            recorder2.stop_recording(data_type);
            disp(['Stopped recording for cycle ', num2str(loop_count), ' (', file_label, ')']);
            % Play ping sound for recording stop
            sound(ping_sound, sound_fs);
        end

        % (Optional) Process recorded files here if needed...

        % Pause between recordings
        disp(['Pausing for ', num2str(pause_duration), ' seconds...']);
        pause(pause_duration);
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
