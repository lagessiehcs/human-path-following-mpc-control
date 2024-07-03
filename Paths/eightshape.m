% Specify waypoints, times of arrival, and sampling rate. 
wp = [0 0 0; 2 -2 0; 4 0 0; 6 2 0; 8 0 0; 6 -2 0; 4 0 0; 2 2 0; 0 0 0];
toa = 10*(0:size(wp,1)-1).';
Fs = 10;
% Create trajectory. 
traj = waypointTrajectory(wp, toa, SampleRate=Fs);
% Get position.
t = 0:1/Fs:toa(end);
pos = lookupPose(traj, t);

% Plot
plot(pos(:,1), pos(:,2))

eight = [pos(:,1), pos(:,2)];

save('eight.mat',"eight")