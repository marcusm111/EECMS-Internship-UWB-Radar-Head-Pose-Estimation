close all;
%clear;

%process_uwb_data(data_file,len,set_range,range_lim,plot_frame_num,offset,fps)
%process_uwb_data('s1p2r1.dat',60,3,1.5,500,3,60,56,0.95);

%{
56
112
224
448
896
%}

%testing_microdop('down.dat', 60, 3, 500,32,0.95);

%five_pos_RT_matrix(data_up,data_down,data_left,data_right,data_nomovement,len,set_range,range_lim,offset,fps)
%five_pos_RT_matrix(12,'up.dat','down.dat','left90.dat','right90.dat','no_movement.dat',60,3,1.3,3,500);

%five_pos_microdop_sig(24,'joseph','up.dat','down.dat','left.dat','right.dat','nomovement.dat',448,0.95,60, 3, 500);

%five_pos_microdop_sig(1,'sensor1','s1up.dat','s1down.dat','s1left.dat','s1right.dat','s1nm.dat',448,0.95,60, 3, 500);
%five_pos_microdop_sig(1,'sensor2','s2up.dat','s2down.dat','s2left.dat','s2right.dat','s2nm.dat',448,0.95,60, 3, 500);

% Directory containing the text files

data_dir = pwd; % Current directory (change if needed)

% Get all .txt files in the directory
data_files = dir(fullfile(data_dir, '*.csv'));
file_names = {data_files.name}; % Extract filenames

% Sort all files into shared p1-p5 folders
sort_data(file_names);


%{
dataFile = fopen('Glasgow_Down.dat','rb');
Data1 = fread(dataFile,'float');
fclose(dataFile);
figure();
plot(abs(Data1(1:1000)));
%}

%frameareaoffset = 0.18
%frameareastart = -0.025761
%frameareastop = 3.0092
%framebincount = 60

%{
len = len/2;
i_vec = data(1:len);
q_vec = data(len+1:len*2);
iq_vec = i_vec + 1i*q_vec;
data = abs(iq_vec);
%}

%num_frames = length(Data)/

%{
index = 1:240;
dataFile = fopen('down.dat','rb');
Data = fread(dataFile,'float');
fclose(dataFile);

%recreating a single radar frame by summing the transmit and receive
%waveforms for the given frame

len = 60;

frame1I = Data(3:3+len);
frame1Q = Data(3+len+1:3+2*len+1);
frame2I = Data(3+3+2*len+1:3+3+3*len+1);
frame2Q = Data(3+3+3*len+1+1:3+3+4*len+1+1);

frame1 = frame1I + 1i*frame1Q;
frame2 = frame2I + 1i*frame2Q;
frame1 = abs(frame1);
frame2 = abs(frame2);
frameindex = 1:61;


figure();
%plot(abs(frame2(frameindex)));
plot(abs(Data(index)));
title('raw data');
xlabel('index(?)');
ylabel('Amplitude');
%}


%{
fileName = 'up.dat'; %change name to match file name of raw data 
fileID = fopen(fileName, 'rb');
if fileID == -1
    error('cannot open file');
end
rawData = fread(fileID,'float');
fclose(fileID);

fileName1 = 'down.dat'; %change name to match file name of raw data 
fileID1 = fopen(fileName1, 'rb');
if fileID1 == - 1
    error('cannot open file');
end
rawData1 = fread(fileID1,'float');
fclose(fileID1);

left45file = fopen('left45.dat','rb');
left45data = fread(left45file,'float');
fclose(left45file);

left90file = fopen('left90.dat','rb');
left90data = fread(left90file,'float');
fclose(left90file)

right45file = fopen('right45.dat','rb');
right45data = fread(right45file,'float');
fclose(right45file);

right90file = fopen('right90.dat','rb');
right90data = fread(right90file,'float');
fclose(right90file)

downlen = [1:length(rawData1)];

downlen = downlen';
%}


%{
%plot raw data
figure();
hold("on");
%plot(rawData(index));
plot(rawData1(index));
plot(left45data(index));
%plot(left90data(index));
plot(right45data(index));
%plot(right90data(index));
%legend('up','down','left45','left90','right45','right90');
legend('up','left45','right45');
%plot(linspace(0,100,800),rawData1(1:800),"LineWidth",0.5);
title('raw data');
xlabel('index(?)');
ylabel('Amplitude');
%}