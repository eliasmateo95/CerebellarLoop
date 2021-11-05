function ComplexPlotsAfterSTDP(Type,Simulation,Data,Params)
if Type == "Coupled"
    Name = Params.Coupled.simul;
    Name = Name + ' ' + string(Simulation);
    Plots = Params.Plot.Coupled;
    SimParams = Params.Coupled;
    SimData = Data.Coupled;
elseif Type == "Uncoupled"
    Name = Params.Uncoupled.simul;
    Name = Name + ' ' + string(Simulation);
    Plots = Params.Plot.Uncoupled;
    SimParams = Params.Uncoupled;
    SimData = Data.Uncoupled;
end
f = waitbar(0,char('Please wait for Complex Plots '+string(Simulation)));
%% CS Triggered Noise 
if Plots.CSTrigNoise == 'y'
    plotCStrigNoise(Name,SimData.IO_spikes,SimData.PC_noise,SimData.IO_Vs,SimData.Noise_t,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f1 = waitbar(1/9,char('CS Triggered Noise Plots Finished '+string(Simulation)));close(f)
% CS Triggered SS
if Plots.CSTrigSS == 'y'
    plotCStrigSS(Name,SimData.IO_spikes,SimData.PC_v,SimData.IO_Vs,SimData.Noise_t,SimData.PC_spikes,SimData.PC_firingrate,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f2 = waitbar(2/9,char('CS Triggered SS Plots Finished '+string(Simulation)));close(f1)
% CS Triggered DCN
if Plots.CSTrigDCN == 'y'
    plotCStrigDCN(Name,SimData.IO_spikes,SimData.IO_Vs,SimData.Noise_t,SimData.DCN_spikes,SimData.DCN_v,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f3 = waitbar(3/9,char('CS Triggered DCN Plots Finished '+string(Simulation)));
close(f2)
% DCN Spikes Triggered Noise 
if Plots.DCNTrigNoise == 'y'
    plotDCNtrigNoise(Name,SimData.DCN_spikes,SimData.PC_noise,SimData.DCN_v,SimData.Noise_t,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f4 = waitbar(4/9,char('DCN Triggered Noise Plots Finished '+string(Simulation)));close(f3)
% Firing Rates
if Plots.firing == 'y'
    firingplots(Name,SimData.PC_firingrate,SimData.DCN_firingrate,SimData.IO_firingrate,SimData.Noise_t,Params.window,Params.save,SimParams.fname,Params.show);
end
f5 = waitbar(5/9,char('Firing Frequency Plots Finished '+string(Simulation)));close(f4)
% Rates Cross-correlated
if Plots.corrfiring == 'y'
    corrfiringplots(Name,SimData.PC_firingrate,SimData.DCN_firingrate,SimData.IO_firingrate,Params.save,SimParams.fname,Params.show);
end
f6 = waitbar(6/9,char('Firing Frequency Correlation Plots Finished '+string(Simulation)));close(f5)
% SS CS Raster Plot
if Plots.raster == 'y'
    rasterplotPCIO(Name,SimData.IO_spikes,SimData.PC_spikes,SimData.IO_Vs,SimData.Noise_t,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f7 = waitbar(7/9,char('Raster Plots Finished '+string(Simulation)));close(f6)
% Waterfall Plot IO Membrane Potentials
if Plots.waterfallIOVs == 'y'
    waterfallIOVs(Name,SimData.IO_spikes,SimData.IO_Vs,SimData.Noise_t,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f10 = waitbar(8/9,char('CS Triggered Noise Plots Finished '+string(Simulation)));close(f7)
% Waterfall Plot Noise
if Plots.waterfallNoise == 'y'
    waterfallNoise(Name,SimData.PC_noise,SimData.IO_spikes,SimData.Noise_t,Params.timepreceding,Params.save,SimParams.fname,Params.show);
end
f11 = waitbar(9/9,char('CS Triggered Noise Plots Finished '+string(Simulation))); close(f10);pause(.5);close(f11);
f12 = waitbar(9/9,char('Complex Plots Finished, Please Close Window '+string(Simulation)));pause(1);close(f12)
end

function plotCStrigNoise(Name,IO_spikes,PC_noise,IO_Vs,Noise_t,timepreceding,save,fname,show)
if iscell(IO_Vs)
    [~,colIO] = size(IO_Vs);
else
    [colIO,~] = size(IO_Vs);
end

spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
% timepreceding = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
PC_noises = cell(colIO,1);
PC_noise_1 = PC_noise(1,:);
PC_noise_2 = PC_noise(2,:);
IOs = cell(colIO,1);
IO_Vs_1 = IO_Vs(1,:);
IO_Vs_2 = IO_Vs(2,:);

for i=1:colIO 
    if isempty(IO_spikes)
        continue
    end
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end    

for i=1:colIO 
    h(i) = figure('visible', show); hold on;
%         timepreceding = 300/0.025;%spikes{i,1}(spiketimes{i,1}(1))/(0.025/1000)-2;
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    PC_noises{i,1}=cell(colsptim-2,1);
    IOs{i,1}=cell(colsptim-2,1);
    PC = PC_noise(i,:);
    IO = IO_Vs(i,:);
    spike_time{i,1} = [];
    counter = 0;
    [~,colPC] = size(PC);
    for ss = 1:colsptim
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colPC
            break
        end
        counter = counter + 1;
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        PC_noises{i,1}{counter,1} = PC(start:finish)/10^-9;
        IOs{i,1}{counter,1} = IO(start:finish)/10^-3;
        if isempty(IOs{i,1}{counter,1})
            continue
        end
        [~,colxax] = size(Noise_t(start:finish));
        xaxis = -colxax/2:colxax/2;
        xaxis = xaxis(2:end).*0.025;
        subplot(2,1,1);hold on;
        plot(xaxis,PC(start:finish)/10^-9);hold on;
        xlabel('t (msecond)')
        ylabel('I (nA)')
        title('Noise Cell '+ string(i)+' '+Name)
        xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
        subplot(2,1,2);hold on;
        plot(xaxis,IO(start:finish)/10^-3);hold on;
        xlabel('t (msecond)')
        ylabel('IO_{Vs} (mV)')
        title('Complex Spikes Cell '+ string(i)+' '+Name)
        xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    end
end    
if save == "True"
saveas(h(1), fullfile(fname,'CS_Noise_1'), 'jpeg');
saveas(h(2), fullfile(fname,'CS_Noise_2'), 'jpeg');
end
h =  figure('Renderer', 'painters', 'visible', show); hold on;
for i=1:colIO 
    [row,~] = size(PC_noises{i,1});
    for ii = 1:row
        [~,col] = size(PC_noises{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(PC_noises{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumPCnoises = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumPCnoises(aa,:) = PC_noises{i,1}{j,1};
    end
    sumPCnoise{i,1} = mean(sumPCnoises);
    
    [row,~] = size(IOs{i,1});
    for ii = 1:row
        [~,col] = size(IOs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(IOs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumIOs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumIOs(aa,:) = IOs{i,1}{j,1};
    end
    sumIO{i,1} = mean(sumIOs); 
    spike_time{i,1} = [];
    for ss = 1:colsptim-2
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        start = int64(spike_time{i,1}(ss)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spike_time{i,1}(ss)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
            continue
        end
        if start > 0
            break
        end
    end
    y = cell2mat(PC_noises{i,1});                       % Create Dependent Variable ‘Experiments’ Data
    [N] = size(y,1);                                    % Number of ‘Experiments’ In Data Set
    yMean = mean(y);                                    % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
    [~,colxax] = size(Noise_t(start:finish));
    xaxis = -colxax/2:colxax/2;
    xaxis = xaxis(1:end-1).*0.025;
    subplot(2,1,(i));hold on;
    yyaxis left
%     plot(xaxis,sumPCnoise{i,1});hold on;
    a(1) = plot(xaxis, yMean); M1 = 'Noise';hold on;                         % Plot Mean Of All Experiments
    plot(xaxis, yCI95+yMean)                            % Plot 95% Confidence Intervals Of All Experiments
    xlabel('t (mseconds)')
    ylabel('I (nA)')
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    subplot(2,1,(i));hold on;
    yyaxis right
    a(2) = plot(xaxis,sumIO{i,1}); M3 = 'IO_{Vs}'; hold on;
    ylabel('V (mV)')
    title('Complex Spike Triggered Noise Averages '+ string(i)+' '+Name)
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    legend(a,{'Noise','IO_{Vs}'});
end
if save == "True"
    saveas(h(:), fullfile(fname,'CS_Triggered_Noise_Averages'), 'jpeg');
end
end

function plotCStrigSS(Name,IO_spikes,PC_v,IO_Vs,Noise_t,PC_spikes,PC_firingrate,timepreceding,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
% timepreceding = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
PC_vs = cell(colIO,1);
PC_v_1 = PC_v(:,1);
PC_v_2 = PC_v(:,2);
IOs = cell(colIO,1);
IO_Vs_1 = IO_Vs(1,:);
IO_Vs_2 = IO_Vs(2,:);
for i=1:colIO 
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end    
    
for i=1:colIO 
  h(i) =  figure('Renderer', 'painters', 'visible', show); hold on;
%         timepreceding{i,1} = spikes{i,1}(spiketimes{i,1}(1))/(0.025/1000)-2;
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    PC_vs{i,1}=cell(colsptim-2,1);
    IOs{i,1}=cell(colsptim-2,1);
    PC = PC_v(i,:);
    PC_spiking = PC_v(i,:);
    colPC = size(PC_spiking,2);
    if iscell(PC_spikes)
        Times_PC = PC_spikes{1,i}/(0.025/1000);
        [~,col] = size(Times_PC);
    else
        Times_PC = PC_spikes(i,:)/(0.025/1000);
        [~,col] = size(Times_PC);
    end
    for t = 1:col
        jj = int64(Times_PC(t)-colPC);
        PC_spiking(jj) = 0.05;
    end
    IO = IO_Vs(i,:);
    spike_time{i,1} = [];
    counter = 0;
    [~,colPC] = size(PC);  
    for ss = 1:colsptim
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colPC
            break
        end
        counter = counter + 1;
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000);
        PC_vs{i,1}{counter,1} = PC_spiking(start:finish)/10^-3;
        IOs{i,1}{counter,1} = IO(start:finish)/10^-3;
        [~,colxax] = size(Noise_t(start:finish));
        xaxis = -colxax/2:colxax/2;
        xaxis = xaxis(1:end-1).*0.025;
        subplot(2,1,1);hold on;
        plot(xaxis,PC_spiking(start:finish)/10^-3);hold on;
        xlabel('t (msecond)')
        ylabel('PC_V (mV)')
        title('PC '+ string(i)+' '+Name)
        xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
        subplot(2,1,2);hold on;
        plot(xaxis,IO(start:finish)/10^-3);hold on;
        xlabel('t (msecond)')
        ylabel('IO_{Vs} (mV)')
        title('IO Cell '+ string(i)+' '+Name)
        xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    end
end 
if save == "True"
  saveas(h(1), fullfile(fname,'IO_PC_1'), 'jpeg'); 
  saveas(h(2), fullfile(fname,'IO_PC_2'), 'jpeg'); 
end
  
    
    
    
h =  figure('Renderer', 'painters', 'visible', show); hold on;
for i=1:colIO 
    [row,~] = size(PC_vs{i,1});
    for ii = 1:row
        [~,col] = size(PC_vs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(PC_vs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumPCvs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumPCvs(aa,:) = PC_vs{i,1}{j,1};
    end
    sumPCv{i,1} = mean(sumPCvs);
    
    [row,~] = size(IOs{i,1});
    for ii = 1:row
        [~,col] = size(IOs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(IOs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumIOs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumIOs(aa,:) = IOs{i,1}{j,1};
    end
    sumIO{i,1} = mean(sumIOs);  
    
    spike_time{i,1} = [];
    for ss = 1:colsptim-2
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        start = int64(spike_time{i,1}(ss)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spike_time{i,1}(ss)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
            continue
        end
        if start > 0 
            break
        end
    end
    [~,colxax] = size(Noise_t(start:finish));
    xaxis = -colxax/2:colxax/2;
    xaxis = xaxis(1:end-1).*0.025;
    subplot(2,1,(i));hold on;
    yyaxis left
    plot(xaxis,sumPCv{i,1});hold on;
    xlabel('t (mseconds)')
    ylabel('PC_v (mV)')
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    subplot(2,1,(i));hold on;
    yyaxis right
    plot(xaxis,sumIO{i,1},'LineWidth',2);hold on;
    ylabel('IO_{Vs} (mV)')
    title('Complex Spike Triggered PC Spikes Averages '+ string(i)+' '+Name)
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    legend('PC_v','IO_{Vs}');
end
if save == "True"
  saveas(h, fullfile(fname,'CS_Triggered_PC_Spike_Averages'), 'jpeg'); 
end

for i=1:colIO 
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    PC_vs1{i,1}=cell(colsptim-2,1);
    IOs{i,1}=cell(colsptim-2,1);
    IO = IO_Vs(i,:);
    spike_time{i,1} = [];
    counter = 0;
    [~,colPC] = size(PC);  
    for ss = 1:colsptim
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colPC
            break
        end
        counter = counter + 1;
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000);
        PC_vs1{i,1}{counter,1} = PC_firingrate(start:finish);
        IOs{i,1}{counter,1} = IO(start:finish)/10^-3;
    end
end

h =  figure('Renderer', 'painters', 'visible', show); hold on;
for i=1:colIO 
    
%     [row,~] = size(PC_vs1{i,1});
%     for ii = 1:row
%         [~,col] = size(PC_vs1{i,1}{ii,1});
%         if col > 0
%             break
%         end
%     end
%     a = 0;
%     index = [];
%     for j = 1:row 
%         if isempty(PC_vs1{i,1}{j,1})
%             a = a+1;
%             index(end+1) = j;
%         end
%     end
%     sumPCvs = [zeros(row-a,col)];
%     range = 1:row;
%     range(index) = [];
%     aa = 0;
%     for j = range 
%         aa = aa+1;
%         sumPCvs(aa,:) = PC_vs1{i,1}{j,1};
%     end
%     sumPCv{i,1} = mean(sumPCvs);
%     
    
    
    [row,~] = size(IOs{i,1});
    for ii = 1:row
        [~,col] = size(IOs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(IOs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumIOs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumIOs(aa,:) = IOs{i,1}{j,1};
    end
    sumIO{i,1} = mean(sumIOs);  
    
    spike_time{i,1} = [];
    for ss = 1:colsptim-2
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        start = int64(spike_time{i,1}(ss)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spike_time{i,1}(ss)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
            continue
        end
        if start > 0 
            break
        end
    end
%     
    y = cell2mat(PC_vs1{i,1});                       % Create Dependent Variable ‘Experiments’ Data
    [N] = size(y,1);                                    % Number of ‘Experiments’ In Data Set
    yMean = mean(y);                                    % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
    [~,colxax] = size(Noise_t(start:finish));
    xaxis = -colxax/2:colxax/2;
    xaxis = xaxis(1:end-1).*0.025;
    subplot(2,1,(i));hold on;
    yyaxis left
%     plot(xaxis,sumPCv{i,1});hold on;
    a(1) = plot(xaxis, yMean); M1 = 'Noise';hold on;                         % Plot Mean Of All Experiments
    plot(xaxis, yCI95+yMean)                            % Plot 95% Confidence Intervals Of All Experiments
    xlabel('t (mseconds)')
    ylabel('PC_{firing rate} (Hz)')
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    subplot(2,1,(i));hold on;
    yyaxis right
    plot(xaxis,sumIO{i,1},'LineWidth',2);hold on;
    ylabel('IO_{Vs} (mV)')
    title('Complex Spike Triggered PC Firing Averages '+ string(i)+' '+Name)
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    legend('PC_{firing rate}','IO_{Vs}');
end
if save == "True"
  saveas(h, fullfile(fname,'CS_Triggered_PC_Firing_Averages'), 'jpeg'); 
end




end

function plotCStrigDCN(Name,IO_spikes,IO_Vs,Noise_t,DCN_spikes,DCN_v,timepreceding,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
[~,colDCN] = size(DCN_spikes);
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
% timepreceding = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
DCNs = cell(colDCN,1);
DCN_v_1 = DCN_v(1,:);
DCN_v_2 = DCN_v(2,:);
IOs = cell(colIO,1);
IO_Vs_1 = IO_Vs(1,:);
IO_Vs_2 = IO_Vs(2,:);
for i=1:colIO 
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end    

    
for i=1:colIO 
%         figure(); hold on;
%         timepreceding{i,1} = spikes{i,1}(spiketimes{i,1}(1))/(0.025/1000)-2;
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    DCNs{i,1}=cell(colsptim-2,1);
    IOs{i,1}=cell(colsptim-2,1);
    DCN = DCN_v(i,:);
    IO = IO_Vs(i,:);
    spike_time{i,1} = [];
    counter = 0;
    [~,colDCN] = size(DCN_v);
    for ss = 1:colsptim
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colDCN
            break
        end
        counter = counter + 1;
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        IOs{i,1}{counter,1} = IO(start:finish)/10^-3;
        DCNs{i,1}{counter,1} = DCN(start:finish)/10^-3;
    end
end 
   
h =  figure('Renderer', 'painters', 'visible', show); hold on;
for i=1:colIO 
    [row,~] = size(DCNs{i,1});
    for ii = 1:row
        [~,col] = size(DCNs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(DCNs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumDCNs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumDCNs(aa,:) = DCNs{i,1}{j,1};
    end
    sumDCN{i,1} = mean(sumDCNs);
    
    
    
    [row,~] = size(IOs{i,1});
    for ii = 1:row
        [~,col] = size(IOs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(IOs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumIOs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumIOs(aa,:) = IOs{i,1}{j,1};
    end
    sumIO{i,1} = mean(sumIOs);  
    
    spike_time{i,1} = [];
    for ss = 1:colsptim-2
        spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
        start = int64(spike_time{i,1}(ss)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spike_time{i,1}(ss)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
            continue
        end
        if start > 0 
            break
        end
    end
    [~,colxax] = size(Noise_t(start:finish));
    xaxis = -colxax/2:colxax/2;
    xaxis = xaxis(1:end-1).*0.025;
    subplot(2,1,(i));hold on;
    yyaxis left
    plot(xaxis,sumDCN{i,1});hold on;
    xlabel('t (mseconds)')
    ylabel('DCN_v (mV)')
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    subplot(2,1,(i));hold on;
    yyaxis right
    plot(xaxis,sumIO{i,1},'LineWidth',2);hold on;
    ylabel('IO_{Vs} (mV)')
    title('Complex Spike Triggered DCN Spikes Averages '+ string(i)+' '+Name)
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    legend('DCN','IO_{Vs}');
end
if save == "True"
  saveas(h, fullfile(fname,'CS_Triggered_DCN_Spike_Averages'), 'jpeg'); 
end
end

function plotDCNtrigNoise(Name,DCN_spikes,PC_noise,DCN_v,Noise_t,timepreceding,save,fname,show)
if iscell(DCN_spikes)
    [~,colDCN] = size(DCN_spikes);
else
    [colDCN,~] = size(DCN_spikes);
end
spikes = cell(colDCN,1);
indexes = cell(colDCN,1);
spiketimes = cell(colDCN,1);
spike_time = cell(colDCN,1);
colxax = cell(colDCN,1);
PC_noises = cell(colDCN,1);
PC_noise_1 = PC_noise(1,:);
PC_noise_2 = PC_noise(2,:);
DCNs = cell(colDCN,1);
DCN_v_1 = DCN_v(1,:);
DCN_v_2 = DCN_v(2,:);
for i=1:colDCN 
    if iscell(DCN_spikes)
    	[~,col] = size(DCN_spikes{1,i});
    else
        [~,col] = size(DCN_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(DCN_spikes)
    	spikes{i,1}(1) = DCN_spikes{1,i}(1);
    else
        spikes{i,1}(1) = DCN_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(DCN_spikes)
            spikes{i,1}(k) = DCN_spikes{1,i}(k);
            if DCN_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = DCN_spikes(i,k);
            if DCN_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end  
for i=1:colDCN 
        [~,colsptim] = size(spiketimes{i,1});
        xaxis = cell(colsptim-2,1);
%         PC_noises{i,1}=cell(colsptim-2,1);
%         DCNs{i,1}=cell(colsptim-2,1);
        PC = PC_noise(i,:);
        DCN = DCN_v(i,:);
        spike_time{i,1} = [];
        [~,colPC] = size(PC);
        counter = 0;
        for ss = 1:colsptim
            spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000); 
            start = int64(spike_time{i,1}(ss)-timepreceding{i,1}-Noise_t(1)/0.025);
            finish = int64(spike_time{i,1}(ss)+timepreceding{i,1}-Noise_t(1)/0.025);
            if start < 0 
                continue 
            end
            if finish > colPC
                continue
            end
            counter = counter + 1;
            PC_noises{i,1}{counter,1} = PC(start:finish)/10^-9;
            DCNs{i,1}{counter,1} = DCN(start:finish)/10^-3;
        end
 end    
h = figure('Renderer', 'painters', 'visible', show); hold on;
for i=1:colDCN 
    [row,~] = size(PC_noises{i,1});
    for ii = 1:row
        [~,col] = size(PC_noises{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(PC_noises{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumPCnoises = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumPCnoises(aa,:) = PC_noises{i,1}{j,1};
    end
    sumPCnoise{i,1} = mean(sumPCnoises);
    
    [row,~] = size(DCNs{i,1});
    for ii = 1:row
        [~,col] = size(DCNs{i,1}{ii,1});
        if col > 0
            break
        end
    end
    a = 0;
    index = [];
    for j = 1:row 
        if isempty(DCNs{i,1}{j,1})
            a = a+1;
            index(end+1) = j;
        end
    end
    sumDCNs = [zeros(row-a,col)];
    range = 1:row;
    range(index) = [];
    aa = 0;
    for j = range 
        aa = aa+1;
        sumDCNs(aa,:) = DCNs{i,1}{j,1};
    end
    sumDCN{i,1} = mean(sumDCNs); 
    spike_time{i,1} = [];
    for ss = 1:colsptim
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
            continue
        end
        if start > 0 
            break
        end
    end
    y = cell2mat(PC_noises{i,1});                       % Create Dependent Variable ‘Experiments’ Data
    [N] = size(y,1);                                    % Number of ‘Experiments’ In Data Set
    yMean = mean(y);                                    % Mean Of All Experiments At Each Value Of ‘x’
    ySEM = std(y)/sqrt(N);                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
    CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
    yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’
    [~,colxax] = size(Noise_t(start:finish));
    xaxis = -colxax/2:colxax/2;
    xaxis = xaxis(1:end-1).*0.025;
    subplot(2,1,(i));hold on;
    yyaxis left
%     plot(xaxis,sumPCnoise{i,1});hold on;
    a(1) = plot(xaxis, yMean); M1 = 'Noise';hold on;                         % Plot Mean Of All Experiments
    plot(xaxis, yCI95+yMean)                            % Plot 95% Confidence Intervals Of All Experiments
    xlabel('t (mseconds)')
    ylabel('I (nA)')
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    subplot(2,1,(i));hold on;
    yyaxis right
    a(2) = plot(xaxis,sumDCN{i,1});hold on;
    ylabel('V (mV)')
    title('DCN Spike Triggered Noise Averages '+ string(i)+' '+Name)
    xlim([-timepreceding{i,1}*0.025  timepreceding{i,1}*0.025])
    legend(a,{'Noise','DCN_v'});
end
if save == "True"
    saveas(h, fullfile(fname,'DCN_Triggered_Noise_Averages'), 'jpeg');
end
end

function firingplots(Name,PC_firingrate,DCN_firingrate,IO_firingrate,Noise_t,window,save,fname,show)
h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(Noise_t/1000,DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(Noise_t/1000,PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('DCN PC Firing Rates '+Name)
subplot(2,1,2)
a1 = plot(Noise_t/1000,DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(Noise_t/1000,PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('DCN PC Firing Rates Window '+Name)
xlim([window(1)/1000+Noise_t(1)/1000 window(2)/1000+Noise_t(1)/1000])
if save == "True"
    saveas(h, fullfile(fname,'DCN_PC_FiringRates'), 'jpeg');
end

h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(Noise_t/1000,IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(Noise_t/1000,PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('IO PC Firing Rates '+Name)
subplot(2,1,2)
a1 = plot(Noise_t/1000,IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(Noise_t/1000,PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('IO PC Firing Rates Window '+Name)
xlim([window(1)/1000+Noise_t(1)/1000 window(2)/1000+Noise_t(1)/1000])
if save == "True"
    saveas(h, fullfile(fname,'IO_PC_FiringRates'), 'jpeg');
end

h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(Noise_t/1000,IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(Noise_t/1000,DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('IO DCN Firing Rates '+Name)
subplot(2,1,2)
a1 = plot(Noise_t/1000,IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(Noise_t/1000,DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title('IO DCN Firing Rates Window '+Name)
xlim([window(1)/1000+Noise_t(1)/1000 window(2)/1000+Noise_t(1)/1000])
if save == "True"
    saveas(h, fullfile(fname,'IO_DCN_FiringRates'), 'jpeg');
end
end

function corrfiringplots(Name,PC_firingrate,DCN_firingrate,IO_firingrate,save,fname,show)

[r,lags] = xcorr(DCN_firingrate,PC_firingrate,'unbiased');
h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
plot(lags*0.025/1000,r)
title('DCN vs. PC Rates Cross-correlated '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
subplot(2,1,2)
plot(lags*0.025/1000,r)
xlim([-1 1])
title('DCN vs. PC Rates Cross-correlated Window '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
if save == "True"
    saveas(h, fullfile(fname,'DCN_PC_Rates_Xcorr'), 'jpeg');
end 

[r,lags] = xcorr(IO_firingrate,DCN_firingrate,'unbiased');
h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
plot(lags*0.025/1000,r)
title('IO vs. DCN Rates Cross-correlated '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
subplot(2,1,2)
plot(lags*0.025/1000,r)
xlim([-1 1])
title('IO vs. DCN Rates Cross-correlated Window '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
if save == "True"
    saveas(h, fullfile(fname,'IO_DCN_Rates_Xcorr'), 'jpeg');
end

[r,lags] = xcorr(IO_firingrate,PC_firingrate,'unbiased');
h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
plot(lags*0.025/1000,r)
title('IO vs. PC Rates Cross-correlated '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
subplot(2,1,2)
plot(lags*0.025/1000,r)
xlim([-1 1])
title('IO vs. PC Rates Cross-correlated Window '+Name)
xlabel('t (seconds)')
ylabel('Cross Correlation')
if save == "True"
    saveas(h, fullfile(fname,'IO_PC_Rates_Xcorr'), 'jpeg');
end

end

function rasterplotPCIO(Name,IO_spikes,PC_spikes,IO_Vs,Noise_t,timepreceding,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
spike_time = cell(colIO,1);
for i=1:colIO 
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
    IO = IO_Vs(i,:);
    IO_spiketimes{i,1} = [];
    for k = 1:size(spiketimes{i,1},2)
        start = spikes{i,1}(spiketimes{i,1}(k))*1000/0.025-Noise_t(1)/0.025;
        if k > (size(spiketimes{i,1},2)-1)
            finish = spikes{i,1}(end)*1000/0.025-Noise_t(1)/0.025;
        else
            finish = spikes{i,1}(spiketimes{i,1}(k+1))*1000/0.025-Noise_t(1)/0.025;
        end
        range_spikes = start:finish;
        [~,I] = max(IO(int64(range_spikes)));
        IO_spiketimes{i,1}{k,1} = range_spikes(I);
    end  
%     
%     [~,colsptim] = size(spiketimes{i,1});
%     spike_time{i,1} = [];
%     counter = 0;
%     for ss = 1:colsptim-2
%         start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1});
%         finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1});
%         if start < 0
%            continue
%         end
%         counter = counter + 1;
%         spike_time{i,1}(end+1) = spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000);
%         spikesstime{i,1}{counter,1} = spikes{i,1}(spiketimes{i,1}(ss))*1000;    
%     end
end    

b = cell(2,1);
for i = 1:colIO
    spikesstime = IO_spiketimes;
    dd = spikesstime{i,1};
    time = timepreceding{i,1};
    [row,~] = size(spikesstime{i,1});
    b{i,1} = cell(row-1,1);
    b{i,1}{1,1} = spikesstime{i,1}(1);
    for j = 2:row
       counter = 0;
       if (dd{j}+time)>dd{j-1}
            counter = counter + 1;
            b{i,1}{j,1}(counter) = dd(j-1);
            b{i,1}{j,1}(counter+1) = dd(j);
       end 
    end
    
    for k = 1:row-1
        [~,col] = size(b{i,1}{k,1});
        if (dd{k}+time)>dd{k+1}
            b{i,1}{k,1}(col+1) = dd(k+1);
        end
    end
end

for j = 1:2
    clear trr1
    spikes = b{j,1}(2:end-1);
    [row,~] = size(spikes);
    tr = cell(row,1);
    for i = 1:row  
        tr{i} = cell2mat(spikes{i,1});
    end 
    rows = cellfun('size',tr,1);
    cols = cellfun('size',tr,2);
    maxcol = max(cols);
    for kk = 1:row
        [~,col2] = size(tr{kk});
        if col2<maxcol
            trr1{kk,1} = [cell2mat(tr(kk)) tr{kk,1}(col2)];
        else
            trr1{kk,1} = cell2mat(tr(kk));
        end
    end
    trr = cell2mat(trr1);
    [row,~] = size(trr);
    h(j) = figure('visible', show); hold on;
    for i = 1:row
%         [~,col] = size(PC_spikes{1,j});
        time = timepreceding{j,1};
        if iscell(PC_spikes)
            PCSpikes = PC_spikes{1,j}*1000/0.025-Noise_t(1)/0.025;
        else
            PCSpikes = PC_spikes(j,:)*1000/0.025-Noise_t(1)/0.025;
        end
        start = find(PCSpikes<(trr(i,2)+Noise_t(1)-time), 1, 'last');
        finish = find(PCSpikes<(trr(i,2)+Noise_t(1)+time), 1, 'last');
        xx = trr(i,:);
        middleElement = xx(ceil(numel(xx)/2));
        x = xx-middleElement;
        x = x*0.025;
        if x(1) < -timepreceding{j,1}*0.025
            x(1) = 0;
        end
        y = i + zeros(size(trr(i,:)));
        plot(x,y,'k.','MarkerSize',10);hold on;
        xx1 = PCSpikes(start:finish);
        middleElement1 = xx1(ceil(numel(xx1)/2));
        xxx1 = xx1-middleElement1;
        xxx1 = xxx1*0.025;
        y1 = i + zeros(size(xx1));
        plot(xxx1,y1,'k.','MarkerSize',3); hold on;
    %    set(gca,'xtick',[])
    %    set(gca,'ytick',[])
    end
    xlabel('t (mseconds)')
    ylabel('Trial')
    xlim([-timepreceding{j,1}*0.025 timepreceding{j,1}*0.025])
    ylim([0 row])
    title('Raster Plot Cell '+string(j)+' '+Name); % set title
end

if save == "True"
    saveas(h(1), fullfile(fname,'RasterPlot1'), 'jpeg');
    saveas(h(2), fullfile(fname,'RasterPlot2'), 'jpeg');
end
end

function waterfallIOVs(Name,IO_spikes,IO_Vs,Noise_t,timepreceding,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
IOs = cell(colIO,1);
IO_Vs_1 = IO_Vs(1,:);
IO_Vs_2 = IO_Vs(2,:);

for i=1:colIO 
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end    

for i=1:colIO 
    [~,colsptim] = size(spiketimes{i,1});
    IOs{i,1}=cell(colsptim-2,1);
    IO = IO_Vs(i,:);
    [~,colIO] = size(IO);
    counter = 0;
    for ss = 1:colsptim-2 
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colIO
            break
        end
        counter = counter + 1;
        IOs{i,1}{counter,1} = IO(start:finish)/10^-3;
        if isempty(IOs{i,1}{counter,1})
            continue
        end
    end
end   

IOs_Noise = IOs;
az = 40;
el = 45;
IO_snipets = cell(2,1);
for i = 1:2
    IO_snipets{i,1} = cell2mat(IOs_Noise{i,1});
    [row,col] = size(IO_snipets{i,1});
    range = -col/2:col/2-1;
    x = range*0.025;
    y = 1:row;
    zz = IO_snipets{i,1};
    [xx,yy] = meshgrid(x,y);
    h(i) = figure('visible', show);
    surf(xx,yy,zz,yy);
    xlabel('time (ms)')
    ylabel('Event')
    zlabel('V (mV)')
    title('Waterfall IO Membrane Potential '+string(i)+' '+Name)
    shading flat
    c = colorbar('SouthOutside');
    c.Label.String = 'Trials';
    colormap(gray)
    view(az, el);
end
if save == "True"
    saveas(h(1), fullfile(fname,'Waterfall_IO_1'), 'jpeg');
    saveas(h(2), fullfile(fname,'Waterfall_IO_2'), 'jpeg');
end
end

function waterfallNoise(Name,PC_noise,IO_spikes,Noise_t,timepreceding,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
PC_noises = cell(colIO,1);
PC_noise_1 = PC_noise(1,:);
PC_noise_2 = PC_noise(2,:);
for i=1:colIO 
    if iscell(IO_spikes)
    	[~,col] = size(IO_spikes{1,i});
    else
        [~,col] = size(IO_spikes(i,:));
    end
    spikes{i,1} = zeros(1,col);
    if iscell(IO_spikes)
    	spikes{i,1}(1) = IO_spikes{1,i}(1);
    else
        spikes{i,1}(1) = IO_spikes(i,1);
    end
    indexes{i,1} = zeros(1,col);
    for k = 2:col
        if iscell(IO_spikes)
            spikes{i,1}(k) = IO_spikes{1,i}(k);
            if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        else
            spikes{i,1}(k) = IO_spikes(i,k);
            if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
               indexes{i,1}(k) = k;
            end
        end
    end
    [~,colindex] = size(indexes{i,1});
    spiketimes{i,1} = [];
    for kk = 1:colindex
       if indexes{i,1}(kk) == 0
           spiketimes{i,1}(end+1) = kk;
       end
    end 
end    

for i=1:colIO 
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    PC_noises{i,1}=cell(colsptim-2,1);
    PC = PC_noise(i,:);
    colPC = size(PC,2);
    counter = 0;
    for ss = 1:colsptim-2
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)-timepreceding{i,1}-Noise_t(1)/0.025);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(0.025/1000)+timepreceding{i,1}-Noise_t(1)/0.025);
        if start < 0
           continue
        end
        if finish > colPC
            break 
        end
        counter = counter + 1;
        PC_noises{i,1}{counter,1} = PC(start:finish)/10^-9;
    end
end    

PC_noises_IO = PC_noises;
az = 40;
el = 45;
Noise_snipets = cell(2,1);
for i = 1:2
    Noise_snipets{i,1} = cell2mat(PC_noises_IO{i,1});
    [row,col] = size(Noise_snipets{i,1});
    range = -col/2:col/2-1;
    x = range*0.025;
    y = 1:row;
    zz = Noise_snipets{i,1};
    [xx,yy] = meshgrid(x,y);
    h(i) = figure('visible', show);
    surf(xx,yy,zz,yy)
    xlabel('t (ms)')
    ylabel('Event')
    zlabel('I (nA)')
    title('Waterfall Noise '+string(i)+' '+Name)
    shading flat
    c = colorbar('SouthOutside');
    c.Label.String = 'Trials';
    colormap('copper')
    view(az, el);
end
if save == "True"
    saveas(h(1), fullfile(fname,'Waterfall_Noise_1'), 'png');
    saveas(h(2), fullfile(fname,'Waterfall_Noise_2'), 'png');
end
end





