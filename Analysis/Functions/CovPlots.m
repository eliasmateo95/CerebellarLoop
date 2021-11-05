function CovPlots(Type,Simulation,Data,Params,Number)
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
if Plots.Noisecov == 'y'
    if Number == 1
        Noisecov1(Name,SimData.IO_spikes,SimData.PC_noise,SimData.Noise_t,Params.timepreceding_Cov,Params.save,SimParams.fname,Params.show,Params.dtime);
    elseif Number == 2
        Noisecov2(Name,SimData.IO_spikes,SimData.PC_noise,SimData.Noise_t,Params.timepreceding_Cov,Params.save,SimParams.fname,Params.show,Params.dtime);
    end
end
f8 = waitbar(1/2,char('Noise Covariance Plots Finished '+string(Simulation)));close(f7);pause(.5);close(f8);
end


function Noisecov1(Name,IO_spikes,PC_noise,Noise_t,timepreceding,save,fname,show,dtime)
[~,colIO] = size(IO_spikes);
[rowPC,~] = size(PC_noise);
PC_noises = cell(colIO,1);
Ms_noise = cell(2,1);
for i = 1:rowPC
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
        start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)-timepreceding{1,1}-Noise_t(1)/dtime);
        finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)+timepreceding{1,1}-Noise_t(1)/dtime);
        if start < 0
           continue
        end
        if finish > colPC
            break
        end
        counter = counter + 1;
        PC_startfinish = PC(start:finish)/10^-9;
        [~,colPCstartfinish]=size(PC(start:finish));
        if ss == 1
            PC_noises{i,1}{counter,1} = PC_startfinish;
            colPC_1 = colPCstartfinish;
        else
            if colPC_1 ~= colPCstartfinish
                PC_noises{i,1}{counter,1} = PC_startfinish(1:colPC_1);
            else
                PC_noises{i,1}{counter,1} = PC_startfinish;
%             colPC_prev = colPCstartfinish;
            end
        end
    end
    PC_snipets = cell2mat(PC_noises{i,1});
    M = corrcoef(PC_snipets);
    Ms_noise{i,1} = M;
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
    title('Covariance Matrix Noise '+string(i)+' '+Name, 'FontSize', 14); % set title
    colormap('jet'); % set the colorscheme
    c = colorbar();
    c.Label.String = 'Covariance';
    filename = 'CovNoise_' + string(i);
    if save == "True"
        saveas(h, fullfile(fname,char(filename)), 'jpg');
    end
end
% if save == "True"
%     saveas(h(1), fullfile(fname,'CovNoise1'), 'jpeg');
%     saveas(h(2), fullfile(fname,'CovNoise2'), 'jpeg');
% end
end

% function Noisecov2(Name,IO_spikes,PC_noise,Noise_t,timepreceding,save,fname,show,dtime)
% if iscell(IO_spikes)
%     [~,colIO] = size(IO_spikes);
% else
%     [colIO,~] = size(IO_spikes);
% end
% PC_noises = cell(colIO,1);
% Ms_noise = cell(2,1);
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
% for i=colIO 
%     [~,colsptim] = size(spiketimes{i,1});
%     xaxis = cell(colsptim-2,1);
%     PC = PC_noise(i,:);
%     counter = 0;
%     [~,colPC] = size(PC);
%     for ss = 1:colsptim-2
%         start = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)-timepreceding{i,1}-Noise_t(1)/dtime);
%         finish = int64(spikes{i,1}(spiketimes{i,1}(ss))/(dtime/1000)+timepreceding{i,1}-Noise_t(1)/dtime);
%         if start < 0
%            continue
%         end
%         if finish > colPC
%             break 
%         end
%         counter = counter + 1;
%         PC_noises{i,1}{counter,1} = PC(start:finish)/10^-9;
%     end
% end
% for i=colIO 
%     PC_snipets = cell2mat(PC_noises{i,1});
%     M = corrcoef(PC_snipets);
%     Ms_noise{i,1} = M;
%     h = figure('visible', show);
%     imagesc(M); % plot the matrix
%     [~,n] = size(M);
%     step = 2000;
%     set(gca, 'XTick', 0:step:n); % center x-axis ticks on bins
%     set(gca, 'YTick', 0:step:n); % center y-axis ticks on bins
%     set(gca, 'XTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set x-axis labels
%     set(gca, 'YTickLabel', (-int64(n/2):step:int64(n/2))*dtime); % set y-axis labels
%     xlabel('t (mseconds)')
%     ylabel('t (mseconds)')
%     title('Covariance Matrix Noise '+string(i)+' '+Name, 'FontSize', 14); % set title
%     colormap('jet'); % set the colorscheme
%     c = colorbar();
%     c.Label.String = 'Covariance';
%     filename = 'CovNoise' + string(i);
%     if save == "True"
%         saveas(h, fullfile(fname,char(filename)), 'jpg');
%     end
% end
% % if save == "True"
% %     saveas(h(1), fullfile(fname,'CovNoise1'), 'jpeg');
% %     saveas(h(2), fullfile(fname,'CovNoise2'), 'jpeg');
% % end
% end

