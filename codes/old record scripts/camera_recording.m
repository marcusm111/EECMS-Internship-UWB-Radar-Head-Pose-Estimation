clear all;close all;clc;

vid = VideoWriter('recording.mp4','MPEG-4');

% Open the ZED
zed = webcam('ZED 2')
% Set video resolution
zed.Resolution = zed.AvailableResolutions{1};
% Get image size
[height width channels] = size(snapshot(zed))

% Create Figure and wait for keyboard interrupt to quit
f = figure('name','ZED 2 camera','keypressfcn','close','windowstyle','modal');
ok = 1;
num_frames = 1;

% Start loop
while ok
      % Capture the current image
      img = snapshot(zed);
      image_left = img(:, 1 : width/2, :);
      open(vid);
      writeVideo(vid,image_left);

      % Display the left and right images
      imshow(image_left);
      title('Image Left');
      drawnow;

      % Check for interrupts
      ok = ishandle(f);
  end

  close(vid);
  % close the camera instance
  clear cam