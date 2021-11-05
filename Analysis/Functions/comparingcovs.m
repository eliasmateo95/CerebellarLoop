function comparingcovs(Simulation,Data_beforeSTDP,Data_AfterSTDP,Params)
[PC_noises_coupled_before] = COV("Coupled",Simulation,Data_beforeSTDP,Params);
[PC_noises_coupled_after] = COV("Coupled",Simulation,Data_AfterSTDP,Params);
for i = 1:2;
    size1 = size(PC_noises_coupled_before{i,1},1);
    size2 = size(PC_noises_coupled_after{i,1},1);
    len = min(size1,size2);
    for jj = 1:len
        PC_noises_coup{i,1}{jj,1} = PC_noises_coupled_after{i,1}{jj}-PC_noises_coupled_before{i,1}{jj};
    end
end
%%
[PC_noises_uncoupled_before] = COV("Uncoupled",Simulation,Data_beforeSTDP,Params);
[PC_noises_uncoupled_after] = COV("Uncoupled",Simulation,Data_AfterSTDP,Params);
for i = 1:2;
    size1 = size(PC_noises_coupled_before{i,1},1);
    size2 = size(PC_noises_coupled_after{i,1},1);
    len = min(size1,size2);
    for jj = 1:len
        PC_noises_uncoup{i,1}{jj,1} = PC_noises_coupled_after{i,1}{jj}-PC_noises_coupled_before{i,1}{jj};
    end
end
%%
save = Params.save;
show = Params.show;
fname = Params.Coupled.fname;
for i=1:2 
    PC_snipets = cell2mat(PC_noises_coup{i,1});
    M = corrcoef(PC_snipets);
    Ms_noise{i,1} = M;
    h = figure('visible', show);
    imagesc(M); % plot the matrix
    [~,n] = size(M);
    step = 2000;
    set(gca, 'XTick', 0:step:n); % center x-axis ticks on bins
    set(gca, 'YTick', 0:step:n); % center y-axis ticks on bins
    set(gca, 'XTickLabel', (-int64(n/2):step:int64(n/2))*0.025); % set x-axis labels
    set(gca, 'YTickLabel', (-int64(n/2):step:int64(n/2))*0.025); % set y-axis labels
    xlabel('t (mseconds)')
    ylabel('t (mseconds)')
    title('Covariance Matrix Noise '+string(i)+' Difference for Coupled Cells', 'FontSize', 14); % set title
    colormap('jet'); % set the colorscheme
    c = colorbar();
    c.Label.String = 'Covariance';
    filename = 'CovNoiseCoupled' + string(i);
    if save == "True"
        saveas(h, fullfile(fname,char(filename)), 'jpg');
    end
end
fname = Params.Uncoupled.fname;
for i=1:2 
    PC_snipets = cell2mat(PC_noises_uncoup{i,1});
    M = corrcoef(PC_snipets);
    Ms_noise{i,1} = M;
    h = figure('visible', show);
    imagesc(M); % plot the matrix
    [~,n] = size(M);
    step = 2000;
    set(gca, 'XTick', 0:step:n); % center x-axis ticks on bins
    set(gca, 'YTick', 0:step:n); % center y-axis ticks on bins
    set(gca, 'XTickLabel', (-int64(n/2):step:int64(n/2))*0.025); % set x-axis labels
    set(gca, 'YTickLabel', (-int64(n/2):step:int64(n/2))*0.025); % set y-axis labels
    xlabel('t (mseconds)')
    ylabel('t (mseconds)')
    title('Covariance Matrix Noise '+string(i)+' Difference for Uncoupled Cells', 'FontSize', 14); % set title
    colormap('jet'); % set the colorscheme
    c = colorbar();
    c.Label.String = 'Covariance';
    filename = 'CovNoiseUncoupled' + string(i);
    if save == "True"
        saveas(h, fullfile(fname,char(filename)), 'jpg');
    end
end
end

function [PC_noises] = COV(Type,Simulation,Data,Params)
if Type == "Coupled"
    Name = Params.Coupled.simul;
    Name = Name + ' ' + string(Simulation);
    SimParams = Params.Coupled;
    SimData = Data.Coupled;
elseif Type == "Uncoupled"
    Name = Params.Uncoupled.simul;
    Name = Name + ' ' + string(Simulation);
    SimParams = Params.Uncoupled;
    SimData = Data.Uncoupled;
end
[PC_noises] = coco(Name,SimData.IO_spikes,SimData.PC_noise,SimData.Noise_t,Params.timepreceding_Cov,Params.save,SimParams.fname,Params.show);
end
function [PC_noises] = coco(Name,IO_spikes,PC_noise,Noise_t,timepreceding,save,fname,show)
[~,colIO] = size(IO_spikes);
PC_noises = cell(colIO,1);
Ms_noise = cell(2,1);
for i=1:2
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
    [~,colsptim] = size(spiketimes{i,1});
    xaxis = cell(colsptim-2,1);
    PC = PC_noise(i,:);
    counter = 0;
    [~,colPC] = size(PC);
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
end