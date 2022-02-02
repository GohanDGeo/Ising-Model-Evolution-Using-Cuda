%Giorgos Koutroumpis, AEM: 9668, geokonkou@ece.auth.gr
%Parallel and Distributed Systems, Project 3, 2022

%Data smoothed out for better visual representation.

%Load data
data =  readtable('plot_data.xlsx');

%Create masks to load each version's data
v0Mask = data.version == 0;
v1Mask = data.version == 1;
v2Mask = data.version == 2;
v3Mask = data.version == 3;

%Load each version's data
datav0 = data(v0Mask,:);
datav1 = data(v1Mask,:);
datav2 = data(v2Mask,:);
datav3 = data(v3Mask,:);

%Plot the sequential implementation
figure(1)
for i=2:2:10
    testData = datav0(datav0.k==i,:);
    plot3(testData.n, testData.k, medfilt1(testData.time,20), 'Color', '#EDB120', 'LineWidth', 1);
    hold on
end
xticks(1000:1500:10000)
xlim([1000 10000])
xlabel('Width n of nxn ising model');
ylim([2 10])
ylabel('Number of steps k');
zlabel('Time (ms)');
legend('v0')

%Plot the parallel implementations
figure(2)
for i=2:2:10
    testData = datav1(datav1.k==i,:);
    plot3(testData.n, testData.k, medfilt1(testData.time, 20), 'red', 'LineWidth', 1);
    hold on    
    testData = datav2(datav2.k==i,:);
    plot3(testData.n, testData.k, medfilt1(testData.time, 20), 'Color','#77AC30','LineWidth', 1);
    hold on    
    testData = datav3(datav3.k==i,:);
    plot3(testData.n, testData.k, medfilt1(testData.time, 20), 'blue','LineWidth', 1);
    hold on
end

legend('v1', 'v2', 'v3')
xticks(1000:1500:10000)
xlim([1000 10000])
xlabel('Width n of nxn ising model');
ylim([2 10])
ylabel('Number of steps k');
zlabel('Time (ms)');