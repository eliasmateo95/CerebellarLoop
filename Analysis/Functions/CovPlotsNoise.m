function CovPlotsNoise(Type,Simulation,Data,Params,Number)
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
f7 = waitbar(0,char('Please wait for Covariance Plots '+string(Simulation)));

%% CS Triggered Noise Covariance Matrix Noise
if Plots.PCcov == 'y'
    if Number == 1
        PCcov1(Name,SimData.IO_spikes,SimData.PC_v,SimData.PC_spikes,SimData.Noise_t,Params.timepreceding_Cov,Params.save,SimParams.fname,Params.show,Params.dtime);
    elseif Number == 2
        PCcov2(Name,SimData.IO_spikes,SimData.PC_v,SimData.PC_spikes,SimData.Noise_t,Params.timepreceding_Cov,Params.save,SimParams.fname,Params.show,Params.dtime);
    end
end
f9 = waitbar(2/2,char('PC Covariance Plots Finished '+string(Simulation)));close(f7)
f11 = waitbar(11/11,char('Covariance Plots Finished '+string(Simulation)));close(f9);pause(.5);close(f11);
end
function PCcov1(Name,IO_spikes,PC_v,PC_spikes,Noise_t,timepreceding,save,fname,show,dtime)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
[~,colPC] = size(PC_spikes);
spikes = cell(colIO,1);
indexes = cell(colIO,1);
spiketimes = cell(colIO,1);
% timepreceding = cell(colIO,1);
spike_time = cell(colIO,1);
colxax = cell(colIO,1);
PC_vs = cell(colIO,1);
PC_v_1 = PC_v(:,1);
PC_v_2 = PC_v(:,2);
for i=1:colPC
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
    PC_vs{i,1}=cell(colsptim-2,1);
    PC = PC_v(i,:);
    PC_spiking = PC_v(i,:);
    if iscell(PC_spikes)
        Times_PC = PC_spikes{1,i}/(dtime/1000);
        [~,col] = size(Times_PC);
    else
        Times_PC = PC_spikes(i,:)/(dtime/1000);
        [~,col] = size(Times_PC);
    end
    for t = 1:col
        jj = int64(Times_PC(t));
        PC_spiking(jj) = 0.05;
    end
    counter = 0;
    for ss = 1:colsptim-2
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)-timepreceding{1,1}-Noise_t(1)/dtime);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)+timepreceding{1,1}-Noise_t(1)/dtime);
        if start < 0
           continue
        end
        counter = counter + 1;
        PC_startfinish = PC_spiking(start:finish)/10^-3;
        [~,colPCstartfinish]=size(PC_spiking(start:finish));
        if ss == 1
            PC_vs{i,1}{counter,1} = PC_startfinish;
            colPC_1 = colPCstartfinish;
        else
            if colPC_1 ~= colPCstartfinish
                PC_vs{i,1}{counter,1} = PC_startfinish(1:colPC_1);
            else
                PC_vs{i,1}{counter,1} = PC_startfinish;
%             colPC_prev = colPCstartfinish;
            end
        end            
    end

    Ms_PC_v = cell(2,1);
        PC_snipets = cell2mat(PC_vs{i,1});
        M = corrcoef(PC_snipets);
        Ms_PC_v{i,1} = M;
        h = figure('visible', show);
        imagesc(M); % plot the matrix
        [~,n] = size(M);
        step = 2000;
        set(gca, 'XTick', 0:step:n); % center x-axis ticks on bins
        set(gca, 'YTick', 0:step:n); % center y-axis ticks on bins
        set(gca, 'XTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set x-axis labels
        set(gca, 'YTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set y-axis labels
        xlabel('t (mseconds)')
        ylabel('t (mseconds)')
        name = 'Covariance Matrix PC ';
        title(name+string(i)+' '+Name, 'FontSize', 14); % set title
        colormap('jet'); % set the colorscheme
        c = colorbar();
        c.Label.String = 'Covariance';
        filename = 'CovPC'+string(i);
        if save == "True"
            saveas(h, fullfile(fname,char(filename)), 'jpg');
        end
    end
% if save == "True"
%     saveas(h(1), fullfile(fname,'CovPC1'), 'jpeg');
%     saveas(h(2), fullfile(fname,'CovPC2'), 'jpeg');
% end
end

% function PCcov2(Name,IO_spikes,PC_v,PC_spikes,Noise_t,timepreceding,save,fname,show,dtime)
% if iscell(IO_spikes)
%     [~,colIO] = size(IO_spikes);
% else
%     [colIO,~] = size(IO_spikes);
% end
% spikes = cell(colIO,1);
% indexes = cell(colIO,1);
% spiketimes = cell(colIO,1);
% % timepreceding = cell(colIO,1);
% spike_time = cell(colIO,1);
% colxax = cell(colIO,1);
% PC_vs = cell(colIO,1);
% PC_v_1 = PC_v(:,1);
% PC_v_2 = PC_v(:,2);
% for i=colIO 
%     if iscell(IO_spikes)
%     	[~,col] = size(IO_spikes{1,i});
%     else
%         [~,col] = size(IO_spikes(i,:));
%     end
%     spikes{i,1} = zeros(1,col);
%     if iscell(IO_spikes)
%     	spikes{i,1}(1) = IO_spikes{1,i}(1);
%     else
%         spikes{i,1}(1) = IO_spikes(i,1);
%     end
%     indexes{i,1} = zeros(1,col);
%     for k = 2:col
%         if iscell(IO_spikes)
%             spikes{i,1}(k) = IO_spikes{1,i}(k);
%             if IO_spikes{1,i}(k)<(spikes{i,1}(k-1)+0.003)
%                indexes{i,1}(k) = k;
%             end
%         else
%             spikes{i,1}(k) = IO_spikes(i,k);
%             if IO_spikes(i,k)<(spikes{i,1}(k-1)+0.003)
%                indexes{i,1}(k) = k;
%             end
%         end
%     end
%     [~,colindex] = size(indexes{i,1});
%     spiketimes{i,1} = [];
%     for kk = 1:colindex
%        if indexes{i,1}(kk) == 0
%            spiketimes{i,1}(end+1) = kk;
%        end
%     end 
% end    
%     
% for i=colIO 
%     [~,colsptim] = size(spiketimes{i,1});
%     xaxis = cell(colsptim-2,1);
%     PC_vs{i,1}=cell(colsptim-2,1);
%     PC = PC_v(i,:);
%     PC_spiking = PC_v(i,:);
%     if iscell(PC_spikes)
%         Times_PC = PC_spikes{1,i}/(dtime/1000);
%         [~,col] = size(Times_PC);
%     else
%         Times_PC = PC_spikes(i,:)/(dtime/1000);
%         [~,col] = size(Times_PC);
%     end
%     for t = 1:col
%         jj = int64(Times_PC(t));
%         PC_spiking(jj) = 0.05;
%     end
%     counter = 0;
%     for ss = 1:colsptim-2
%         start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)-timepreceding{i,1}-Noise_t(1)/dtime);
%         finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)+timepreceding{i,1}-Noise_t(1)/dtime);
%         if start < 0
%            continue
%         end
%         counter = counter + 1;
%         PC_vs{i,1}{counter,1} = PC_spiking(start:finish)/10^-3;
%     end
% end
% 
% Ms_PC_v = cell(2,1);
%     for i = colIO
%         PC_snipets = cell2mat(PC_vs{i,1});
%         M = corrcoef(PC_snipets);
%         Ms_PC_v{i,1} = M;
%         h = figure('visible', show);
%         imagesc(M); % plot the matrix
%         [~,n] = size(M);
%         step = 2000;
%         set(gca, 'XTick', 0:step:n); % center x-axis ticks on bins
%         set(gca, 'YTick', 0:step:n); % center y-axis ticks on bins
%         set(gca, 'XTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set x-axis labels
%         set(gca, 'YTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set y-axis labels
%         xlabel('t (mseconds)')
%         ylabel('t (mseconds)')
%         name = 'Covariance Matrix PC ';
%         title(name+string(i)+' '+Name, 'FontSize', 14); % set title
%         colormap('jet'); % set the colorscheme
%         c = colorbar();
%         c.Label.String = 'Covariance';
%         filename = 'CovPC'+string(i);
%         if save == "True"
%             saveas(h, fullfile(fname,char(filename)), 'jpg');
%         end
%     end
% % if save == "True"
% %     saveas(h(1), fullfile(fname,'CovPC1'), 'jpeg');
% %     saveas(h(2), fullfile(fname,'CovPC2'), 'jpeg');
% % end
% end
