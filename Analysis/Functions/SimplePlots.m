function SimplePlots(Type,Simulation,Data,Params)
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

if Type == "Coupled"
    if Plots.IODCNPC == 'y' || Plots.PC == 'y' || Plots.DCN == 'y' || Plots.IO == 'y' 
        [Data] = DCNPCplotstools(SimParams.simul,SimData.PC_v,SimData.PC_spikes,SimData.DCN_v,SimData.DCN_spikes,Data);
        SimData = Data.Coupled;
    end
elseif Type == "Uncoupled"
    if Plots.IODCNPC == 'y' || Plots.PC == 'y' || Plots.DCN == 'y' || Plots.IO == 'y' 
        [Data] = DCNPCplotstools(SimParams.simul,SimData.PC_v,SimData.PC_spikes,SimData.DCN_v,SimData.DCN_spikes,Data);
        SimData = Data.Uncoupled;
    end
end
%%
f = waitbar(0,char('Please wait for Simple Plots '+string(Simulation)));
%% Noise Plot
if Plots.Noise == 'y';
    plotNoise(Name,SimData.Noise_I,SimData.Noise_t,Params.save,SimParams.fname,Params.show);
end
f1 = waitbar(1/7,char('Noise Plots Finished '+string(Simulation)));close(f)
%% Noise on Each PC
if Plots.PCNoise == 'y';
    plotPCNoise(Name,SimData.PC_noise,SimData.Noise_t,Params.save,SimParams.fname,Params.show);
end
f2 = waitbar(2/7,char('PC Noise Plots Finished '+string(Simulation)));close(f1)
%% Plot All Noises
if Plots.AllNoise == 'y';
    plotAllNoise(Name,SimData.Noise_I,SimData.PC_noise,SimData.Noise_t,Params.save,SimParams.fname,Params.show);
end
f3 = waitbar(3/7,char('All Noise Plots Finished '+string(Simulation)));close(f2)
%% IO DCN PC Plot
if Plots.IODCNPC == 'y';
    IODCNPCplot(Name,SimData.PC_spikes,SimData.PC_spikings,SimData.DCN_spikings,SimData.IO_Vs,SimData.Noise_t,Params.window,Params.save,SimParams.fname,Params.show);
end
f4 = waitbar(4/7,char('IO DCN PC Plots Finished '+string(Simulation)));close(f3)
%% PC Plots
if Plots.PC == 'y';
    PCplot(Name,SimData.PC_spikes,SimData.PC_v,SimData.Noise_t,SimData.PC_spikings,Params.window,Params.save,SimParams.fname,Params.show);
end
f5 = waitbar(5/7,char('PC Plots Finished '+string(Simulation)));close(f4)
%% DCN Plots
if Plots.DCN == 'y';
    DCNplot(Name,SimData.DCN_spikes,SimData.Noise_t,SimData.DCN_spikings,Params.window,Params.save,SimParams.fname,Params.show);
end
f6 = waitbar(6/7,char('DCN Plots Finished '+string(Simulation)));close(f5)
%% IO Plots
if Plots.IO == 'y';
    IOplot(Name,SimData.DCN_spikes,SimData.Noise_t,SimData.IO_Vs,Params.window,Params.save,SimParams.fname,Params.show);
end
f7 = waitbar(7/7,char('IO Plots Finished '+string(Simulation)));close(f6);pause(.5);close(f7)
f8 = waitbar(7/7,char('Simple Plots Finished '+string(Simulation)));pause(1);close(f8)
end

function plotNoise(Name,Noise_I,Noise_t,save,fname,show)
    [rownoise,~] = size(Noise_I);
    names = strings(1,rownoise);
    h = figure('Renderer', 'painters', 'visible', show );hold on;
    for i = 1:rownoise
        Noise_plot = squeeze(Noise_I);
        a(i) = plot(Noise_t/1000,Noise_plot(i,:)/10^-9); 
        M(i) = 'Noise '+string(i);
        hold on;
    end
    title('Noise Inputs '+Name)
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('I (nA)')
    if save == "True"
        filename = 'Noise_Inputs';
        saveas(h,fullfile(fname, filename), 'jpeg')
    end
end 

function plotPCNoise(Name,PC_noise,Noise_t,save,fname,show)
    [rowPCnoise,~] = size(PC_noise);
    names = strings(1,rowPCnoise);
    h = figure('Renderer', 'painters', 'visible', show );
    for i = 1:rowPCnoise
        a(i) = plot(Noise_t/1000,PC_noise(i,:)/10^-9); 
        M(i) = 'Postsynaptic Current PC'+string(i);
        hold on;
    end
%     title('PC Noise Inputs '+Name)
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('I (nA)')
    if save == "True"
        saveas(h, fullfile(fname,'PC_Noise_Inputs'),'jpg')
    end
end 

function plotAllNoise(Name,Noise_I,PC_noise,Noise_t,save,fname,show)
    [rownoise,~] = size(Noise_I);
    [rowPCnoise,~] = size(PC_noise);
    names = strings(1,rownoise);
    colors = ['b';'r'];   
    h = figure('Renderer', 'painters', 'visible', show );
    for i = 1:rowPCnoise
%         iii = mod(ii - 1, 10) + 1;
%         if iii == 1
%         end
%         subplot(rowPCnoise/2,2,ii,'Parent', h);
        title('PC Noise Inputs '+Name)
        a(i) = plot(Noise_t/1000,PC_noise(i,:)/10^-9); M(i) = 'Noise PC'+string(i); hold on;
    end
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('I (nA)')
    if save == "True"
        saveas(h, fullfile(fname,'Noises_PC_Noise_All'), 'jpeg');
    end
%     Noise_plot = squeeze(Noise_I);
%     for ii = 1:rowPCnoise
% %         iii = mod(ii - 1, 10) + 1;
% %         if iii == 1
%         h = figure('Renderer', 'painters', 'visible', show );
% %         end
% %         subplot(rowPCnoise/2,2,(ii),'Parent', h);
%         aPC(ii) = plot(Noise_t/1000,PC_noise(ii,:)/10^-9); MPC(ii) = 'Noise PC'+string(ii); hold on;
%         for i = 1:rownoise
%             a(i) = plot(Noise_t/1000,Noise_plot(i,:)/10^-9); M(i) = 'Noise '+string(i); hold on;
%         end
%         title('PC Noise Inputs '+string(ii)+' '+Name)
%         legend([aPC(ii); a(:)], [MPC(ii); M(:)]);
%         xlabel('t (seconds)')
%         ylabel('I (nA)')
%     end
%     if save == "True"
%         saveas(h, fullfile(fname,'Noises_PC_Noise_Both'), 'jpeg');
%     end
end

function [Data] = DCNPCplotstools(simul,PC_v,PC_spikes,DCN_v,DCN_spikes,Data)
if iscell(PC_spikes)
    [~,colIO] = size(PC_spikes);
else
    [colIO,~] = size(PC_spikes);
end
PC_spikings = cell(colIO,1);
DCN_spikings = cell(colIO,1);
for i = 1:colIO
    PC_spiking = PC_v(i,:);
    if iscell(PC_spikes)
        Times_PC = PC_spikes{1,i}/(0.025/1000);
        [~,col] = size(Times_PC);
    else
        Times_PC = PC_spikes(i,:)/(0.025/1000);
        [~,col] = size(Times_PC);
    end
    for t = 1:col
        jj = int64(Times_PC(t));
        PC_spiking(jj) = 0.05;
    end
    PC_spikings{i,1} = PC_spiking;
    DCN_spiking = DCN_v(i,:);
    if iscell(DCN_spikes)
        Times_DCN = DCN_spikes{1,i}/(0.025/1000);
        [~,col] = size(Times_DCN);
    else
        Times_DCN = DCN_spikes(1,i)/(0.025/1000);
        [~,col] = size(Times_DCN);
    end
    for t = 1:col
        jj = int64(Times_DCN(t));
        DCN_spiking(jj) = 0.05;
    end
    DCN_spikings{i,1} = DCN_spiking;
end
if simul == "Coupled"
    Data.Coupled.PC_spikings = PC_spikings;
    Data.Coupled.DCN_spikings = DCN_spikings;
end
if simul == "Uncoupled"
    Data.Uncoupled.PC_spikings = PC_spikings;
    Data.Uncoupled.DCN_spikings = DCN_spikings;
end
end

function IODCNPCplot(Name,PC_spikes,PC_spikings,DCN_spikings,IO_Vs,Noise_t,window,save,fname,show)
if iscell(PC_spikes)
    [~,colIO] = size(PC_spikes);
else
    [colIO,~] = size(PC_spikes);
end
colors = ['b','k','r'];
for i = 1:colIO
%     iii = mod(i - 1, 10) + 1;
%     if iii == 1
    h = figure('Renderer', 'painters', 'visible', show );
%     end
    noisee = 0:0.025:120000;
%     subplot(colIO/2,2,(i),'Parent', h);
    [~,collPC] = size(PC_spikings{i,1});
    [~,collDCN] = size(DCN_spikings{i,1});
    a(1) = plot(noisee(1:collPC)/1000,PC_spikings{i,1}*1000,colors(1)); M(1) = 'PC '+string(i);hold on;
    a(2) = plot(noisee(1:collDCN)/1000,DCN_spikings{i,1}*1000,colors(2),'LineWidth',2);M(2) = 'DCN '+string(i);hold on;
    a(3) = plot(Noise_t/1000,IO_Vs(i,:)*1000,colors(3),'LineWidth',2); M(3) = 'IO '+string(i);
    title('Cell Responses '+string(i)+' '+Name)
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')    

    if save == "True"
        saveas(h, fullfile(fname,char('Cell_Responses_'+string(i))), 'jpeg');
    end
end

start = int64(window(1)/0.025);
finish = int64(window(2)/0.025);
if iscell(PC_spikes)
    [~,colIO] = size(PC_spikes);
else
    [colIO,~] = size(PC_spikes);
end

for i = 1:colIO
%     iii = mod(i - 1, 10) + 1;
%     if iii == 1
    h = figure('Renderer', 'painters', 'visible', show );
%     end
%     subplot(colIO/2,2,(i),'Parent', h);
%     [~,collPC] = size(PC_spikings{i,1});
%     [~,collDCN] = size(DCN_spikings{i,1}(start:finish));
    a(1) = plot(Noise_t(start:finish)/1000,PC_spikings{i,1}(start:finish)*1000,colors(1)); M(1) = 'PC '+string(i);hold on;
    a(2) = plot(Noise_t(start:finish)/1000,DCN_spikings{i,1}(start:finish)*1000,colors(2),'LineWidth',2);M(2) = 'DCN '+string(i);hold on;
    a(3) = plot(Noise_t(start:finish)/1000,IO_Vs(i,(start:finish))*1000,colors(3),'LineWidth',2); M(3) = 'IO '+string(i);
    title('Cell Responses '+string(i)+' '+Name+' Window')
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
%     xlim([window(1)/1000 window(2)/1000])
    if save == "True"
        saveas(h, fullfile(fname,char('Cell_Responses_Window_'+string(i))), 'jpeg');
    end
end

end

function PCplot(Name,PC_spikes,PC_v,Noise_t,PC_spikings,window,save,fname,show)
if iscell(PC_spikes)
    [~,colIO] = size(PC_spikes);
else
    [colIO,~] = size(PC_spikes);
end
h = figure('Renderer', 'painters', 'visible', show ); hold on;
noisee = 0:0.025:120000;
for i = 1:colIO
[~,collPC] = size(PC_spikings{i,1});
a(i) = plot(noisee(1:collPC)/1000,PC_spikings{i,1}*1000); M(i) = 'PC '+string(i);hold on;
title('PC Responses '+Name)
legend([a(:)], [M(:)]);
xlabel('t (seconds)')
ylabel('V (mV)')
end

if save == "True"
    saveas(h, fullfile(fname,'PC_Responses'), 'jpeg');
end

colors = ['k','r'];
for i = 1:colIO
%     iii = mod(i - 1, 10) + 1;
%     if iii == 1
    h = figure('Renderer', 'painters', 'visible', show );
%     end
%     subplot(colIO/2,2,(i),'Parent', h);
    [~,collPC] = size(PC_spikings{i,1});
    a(i) = plot(noisee(1:collPC)/1000,PC_spikings{i,1}*1000); M(i) = 'PC '+string(i);hold on;
    legend([a(i)], [M(i)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
    title('PC Response '+string(i)+' '+Name)
    if save == "True"
        saveas(h, fullfile(fname,char('PC_Responses_Both_'+string(i))), 'jpeg');
    end
end





start = int64(window(1)/0.025);
finish = int64(window(2)/0.025);
collors = ['-k','-r'];
for i = 1:colIO
h = figure('Renderer', 'painters', 'visible', show ); hold on;
a(i) = plot(Noise_t(start:finish)/1000,PC_spikings{i,1}(start:finish)*1000); M(i) = 'PC '+string(i);hold on;
% plot(Noise_t/1000,PC_v(i,:)*1000);
title('PC Responses '+Name+' Window')
legend([a(:)], [M(:)]);
xlabel('t (seconds)')
ylabel('V (mV)')
% xlim([window(1)/1000 window(2)/1000])

if save == "True"
    saveas(h, fullfile(fname,char('PC_Responses_Window_'+string(i))), 'jpeg');
end
end

% colors = ['k','r'];
% for i = 1:colIO
% %     iii = mod(i - 1, 10) + 1;
% %     if iii == 1
%     h = figure('Renderer', 'painters', 'visible', show );
% %     end
% %     subplot(colIO/2,2,(i),'Parent', h);
%     a(i) = plot(Noise_t(start:finish)/1000,PC_spikings{i,1}(start:finish)*1000); M(i) = 'PC '+string(i);hold on;
%     legend([a(i)], [M(i)]);
%     xlabel('t (seconds)')
%     ylabel('V (mV)')
%     title('PC Response '+string(i)+' '+Name+' Window')
% %     xlim([window(1)/1000 window(2)/1000])
%     if save == "True"
%         saveas(h, fullfile(fname,char('PC_Responses_Both_Window_'+string(i))), 'jpeg');
%     end
% end
end

function DCNplot(Name,DCN_spikes,Noise_t,DCN_spikings,window,save,fname,show)
if iscell(DCN_spikes)
    [~,colIO] = size(DCN_spikes);
else
    [colIO,~] = size(DCN_spikes);
end
noisee = 0:0.025:120000;
for i = 1:length(DCN_spikings)
    [~,collDCN] = size(DCN_spikings{i,1});
%     iii = mod(i - 1, 10) + 1;
%     if iii == 1
    h = figure('Renderer', 'painters', 'visible', show );
%         ii = 1;
%         clear a M
%     end
    a(i) = plot(noisee(1:collDCN)/1000,DCN_spikings{i,1}*1000); M(i) = 'DCN '+string(i);hold on;
    title('DCN Responses '+Name)
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
%     ii = ii+1;
%     if iii == 10
    if save == "True"
        saveas(h, fullfile(fname,sprintf('DCN_Responses%d',i)), 'jpeg');
    end
%     end
end

% colors = ['k','r'];
% for i = 1:length(DCN_spikings)
% %     iii = mod(i - 1, 10) + 1;
% %     if iii == 1
%     h = figure('Renderer', 'painters', 'visible', show );
% %         ii = 1;
% %         clear a M
% %     end
% %     subplot(length(DCN_spikings)/2,2,(i),'Parent', h);
%     [~,collDCN] = size(DCN_spikings{i,1});
%     a(i) = plot(noisee(1:collDCN)/1000,DCN_spikings{i,1}*1000); M(i) = 'DCN '+string(i);hold on;
%     title('DCN Response '+string(i)+' '+Name)
%     legend([a(i)], [M(i)]);
%     xlabel('t (seconds)')
%     ylabel('V (mV)')
% %     ii = ii+1;
% %     if iii == 10
%     if save == "True"
%         saveas(h, fullfile(fname,sprintf('DCN_Responses_Both%d',i)), 'jpeg');
%     end
% %     end
% end



start = int64(window(1)/0.025);
finish = int64(window(2)/0.025);
for i = 1:length(DCN_spikings)
    h= figure('Renderer', 'painters', 'visible', show ); hold on;
    a(i) = plot(Noise_t(start:finish)/1000,DCN_spikings{i,1}(start:finish)*1000); M(i) = 'DCN '+string(i);hold on;
    title('DCN Responses '+Name+' Window')
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
%     xlim([window(1)/1000 window(2)/1000])
    if save == "True"
        saveas(h, fullfile(fname,char('DCN_Responses_Window_'+string(i))), 'jpeg');
    end
end

% colors = ['k','r'];
% for i = 1:length(DCN_spikings)
% %     iii = mod(i - 1, 10) + 1;
% %     if iii == 1
%     h = figure('Renderer', 'painters', 'visible', show );
% %     end
% %     subplot(length(DCN_spikings)/2,2,(i),'Parent', h);
%     a(i) = plot(Noise_t(start:finish)/1000,DCN_spikings{i,1}(start:finish)*1000); M(i) = 'DCN '+string(i);hold on;
%     title('DCN Response '+string(i)+' '+Name+' Window')
%     legend([a(i)], [M(i)]);
%     xlabel('t (seconds)')
%     ylabel('V (mV)')
% %     xlim([window(1)/1000 window(2)/1000])
%     if save == "True"
%         saveas(h, fullfile(fname,char('DCN_Responses_Both_Window_'+string(i))), 'jpeg');
%     end
% end
end

function IOplot(Name,IO_spikes,Noise_t,IO_Vs,window,save,fname,show)
if iscell(IO_spikes)
    [~,colIO] = size(IO_spikes);
else
    [colIO,~] = size(IO_spikes);
end
h = figure('Renderer', 'painters', 'visible', show ); hold on;
for i = 1:colIO
    a(i) = plot(Noise_t/1000,IO_Vs(i,:)*1000); M(i) = 'IO '+string(i);
    title('IO Cell Responses '+Name)
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
end
if save == "True"
    saveas(h, fullfile(fname,'IO_Responses'), 'jpeg');
end

colors = ['k','r'];
for i = 1:colIO
    h = figure('Renderer', 'painters', 'visible', show ); hold on;
%     subplot(colIO/2,2,(i),'Parent', h);
    a = plot(Noise_t/1000,IO_Vs(i,:)*1000); M = 'IO '+string(i);hold on;
    title('IO Response '+string(i)+' '+Name)
    legend(a, M);
    xlabel('t (seconds)')
    ylabel('V (mV)')
    if save == "True"
        saveas(h, fullfile(fname,char('IO_Responses_'+string(i))), 'jpeg');
    end
end





start = int64(window(1)/0.025);
finish = int64(window(2)/0.025);
for i = 1:colIO
    h = figure('Renderer', 'painters', 'visible', show ); hold on;
    a = plot(Noise_t(start:finish)/1000,IO_Vs(i,(start:finish))*1000,'LineWidth',2); M = 'IO '+string(i);
    title('IO Cell Responses '+Name+' Window')
    legend(a, M);
    xlabel('t (seconds)')
    ylabel('V (mV)')
    % xlim([window(1)/1000 window(2)/1000])
    if save == "True"
        saveas(h, fullfile(fname,char('IO_Responses_Window_'+string(i))), 'jpeg');
    end
end

colors = ['k','r'];
h = figure('Renderer', 'painters', 'visible', show ); hold on;
title('IO Responses All '+Name+' Window')
for i = 1:colIO
    a(i) = plot(Noise_t(start:finish)/1000,IO_Vs(i,(start:finish))*1000); M(i) = 'IO '+string(i);hold on;
end
legend([a(:)], [M(:)]);
xlabel('t (seconds)')
ylabel('V (mV)')
if save == "True"
    saveas(h, fullfile(fname,'IO_Responses_All_Window'), 'jpeg');
end
end


