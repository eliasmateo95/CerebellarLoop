function [Statistic] = Statistics(Type,Data,Params,Statistic)
timepreceding = Params.timepreceding;
if Type == "Coupled"
    IO_spikes = Data.Coupled.IO_spikes;
    PC_spikes = Data.Coupled.PC_spikes;
    Noise_t = Data.Coupled.Noise_t;
    IO_Vs = Data.Coupled.IO_Vs;
    Sim_Data = Data.Coupled;
elseif Type == "Uncoupled"
    IO_spikes = Data.Uncoupled.IO_spikes;
    PC_spikes = Data.Uncoupled.PC_spikes;
    Noise_t = Data.Uncoupled.Noise_t;
    IO_Vs = Data.Uncoupled.IO_Vs;
    Sim_Data = Data.Uncoupled;
end

%%
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
    indexes{2,1} = zeros(1,col);
    for k = 1:col
        if k == 1
            if iscell(IO_spikes)
                spikes{i,1}(1) = IO_spikes{1,i}(1);
            else
                spikes{i,1}(1) = IO_spikes(i,1);
            end
        end
        if k > 1
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

for j = 1:colIO
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
    for i = 1:row
        xx = trr(i,:);
        middleElement = xx(ceil(numel(xx)/2));
        x = xx-middleElement;
        x = x*0.025;
        if x(1) < -timepreceding{j,1}*0.025
            x(1) = 0;
        end
        spikematrix{1,j}{i,1} = x;  
        y = i + zeros(size(trr(i,:)));
    end
end

IO_amplitude = [];
for i = 1:colIO
    IO = IO_Vs(i,:);
    times = IO_spiketimes{i,1};
    [row,~] = size(times);
    for j = 1:row
        IO_amplitude(j,i) = IO(int64(times{j}));
    end
end


variance = [];
meann = [];
difference = [];
interquar = [];
meanabsdev = [];
rang = [];
standard = [];
dist_prev = [];
dist_post = [];
for i = 1:colIO
    [row,~] = size(spikematrix{1,i});
    for j = 1:row
        A = spikematrix{1,i}{j,1};
%         variance(j,i) = var(A);
%         meann(j,i) = mean(A);
%         interquar(j,i) = iqr(A);
%         meanabsdev(j,i) = mad(A);
%         rang(j,i) = range(A);
%         standard(j,i) = std(A);
        dist_prev(j,i) = A(2)-A(1);
        dist_post(j,i) = A(3)-A(2);
    end
end


    
if Type == "Coupled"
    Statistic.Coupled = struct;
    Statistic.Coupled.dist_prev = dist_prev;
    Statistic.Coupled.dist_post = dist_post;
    Statistic.Coupled.mean_prev = mean(dist_prev);
    Statistic.Coupled.mean_post = mean(dist_post);
    Statistic.Coupled.std_prev = std(dist_prev);
    Statistic.Coupled.std_post = std(dist_post);
    Statistic.Coupled.mad_prev = mad(dist_prev);
    Statistic.Coupled.mad_post = mad(dist_post);
    Statistic.Coupled.median_prev = median(dist_prev);
    Statistic.Coupled.median_post = median(dist_post);
    Statistic.Coupled.IO_amplitude = IO_amplitude;
    Statistic.Coupled.IO_amplitude_mean = mean(IO_amplitude);
    Statistic.Coupled.IO_amplitude_std = std(IO_amplitude);
    Statistic.Coupled.IO_amplitude_mad = mad(IO_amplitude);
    Statistic.Coupled.IO_amplitude_median = median(IO_amplitude);
    Statistic.Coupled.firing_IO_mean = mean(Sim_Data.IO_firingrate);
    Statistic.Coupled.firing_IO_std = std(Sim_Data.IO_firingrate);
    Statistic.Coupled.firing_IO_median = median(Sim_Data.IO_firingrate);
    Statistic.Coupled.firing_IO_mad = mad(Sim_Data.IO_firingrate);
    Statistic.Coupled.firing_PC_mean = mean(Sim_Data.PC_firingrate);
    Statistic.Coupled.firing_PC_std = std(Sim_Data.PC_firingrate);
    Statistic.Coupled.firing_PC_median = median(Sim_Data.PC_firingrate);
    Statistic.Coupled.firing_PC_mad = mad(Sim_Data.PC_firingrate);
    Statistic.Coupled.firing_DCN_mean = mean(Sim_Data.DCN_firingrate);
    Statistic.Coupled.firing_DCN_std = std(Sim_Data.DCN_firingrate);
    Statistic.Coupled.firing_DCN_median = median(Sim_Data.DCN_firingrate);
    Statistic.Coupled.firing_DCN_mad = mad(Sim_Data.DCN_firingrate);
elseif Type == "Uncoupled"
    Statistic.Uncoupled = struct;
    Statistic.Uncoupled.dist_prev = dist_prev;
    Statistic.Uncoupled.dist_post = dist_post;
    Statistic.Uncoupled.mean_prev = mean(dist_prev);
    Statistic.Uncoupled.mean_post = mean(dist_post);
    Statistic.Uncoupled.std_prev = std(dist_prev);
    Statistic.Uncoupled.std_post = std(dist_post);
    Statistic.Uncoupled.mad_prev = mad(dist_prev);
    Statistic.Uncoupled.mad_post = mad(dist_post);
    Statistic.Uncoupled.median_prev = median(dist_prev);
    Statistic.Uncoupled.median_post = median(dist_post);
    Statistic.Uncoupled.IO_amplitude = IO_amplitude;
    Statistic.Uncoupled.IO_amplitude_mean = mean(IO_amplitude);
    Statistic.Uncoupled.IO_amplitude_std = std(IO_amplitude);
    Statistic.Uncoupled.IO_amplitude_mad = mad(IO_amplitude);
    Statistic.Uncoupled.IO_amplitude_median = median(IO_amplitude);
    Statistic.Uncoupled.firing_IO_mean = mean(Sim_Data.IO_firingrate);
    Statistic.Uncoupled.firing_IO_std = std(Sim_Data.IO_firingrate);
    Statistic.Uncoupled.firing_IO_median = median(Sim_Data.IO_firingrate);
    Statistic.Uncoupled.firing_IO_mad = mad(Sim_Data.IO_firingrate);
    Statistic.Uncoupled.firing_PC_mean = mean(Sim_Data.PC_firingrate);
    Statistic.Uncoupled.firing_PC_std = std(Sim_Data.PC_firingrate);
    Statistic.Uncoupled.firing_PC_median = median(Sim_Data.PC_firingrate);
    Statistic.Uncoupled.firing_PC_mad = mad(Sim_Data.PC_firingrate);
    Statistic.Uncoupled.firing_DCN_mean = mean(Sim_Data.DCN_firingrate);
    Statistic.Uncoupled.firing_DCN_std = std(Sim_Data.DCN_firingrate);
    Statistic.Uncoupled.firing_DCN_median = median(Sim_Data.DCN_firingrate);
    Statistic.Uncoupled.firing_DCN_mad = mad(Sim_Data.DCN_firingrate);
end
end