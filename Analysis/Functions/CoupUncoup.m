function CoupUncoup(Data,Params,Simulation)
f = waitbar(0,'Please wait for Coupled/Uncoupled Plots');
if iscell(Data.Coupled.IO_Vs)
    [~,colIO] = size(Data.Coupled.IO_Vs);
else
    [colIO,~] = size(Data.Coupled.IO_Vs);
end
for i = 1:colIO
    h(i) = figure('Renderer', 'painters', 'visible', Params.show); hold on;
    a(1) = plot(Data.Coupled.Noise_t/1000,Data.Coupled.IO_Vs(i,:)*1000); M(1) = 'IO '+string(i)+' Coupled';
    a(2) = plot(Data.Uncoupled.Noise_t/1000,Data.Uncoupled.IO_Vs(i,:)*1000); M(2) = 'IO '+string(i)+' Uncoupled';
    title('IO Cell Responses '+string(i)+' '+char(Simulation))
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
end

if Params.save == "True"
    for i=1:colIO
        saveas(h(i), fullfile(Params.Both.fname,sprintf('IO_Responses_%d',i)), 'jpeg');
    end
end

colors = ['k','r'];
for i = 1:colIO
    h(i) = figure('Renderer', 'painters', 'visible', Params.show); hold on;
    subplot(2,1,1)
    a(1) = plot(Data.Coupled.Noise_t/1000,Data.Coupled.IO_Vs(i,:)*1000,colors(1)); M(1) = 'IO '+string(i)+' Coupled';hold on;
    title('IO Response '+string(i)+' Coupled'+' '+char(Simulation))
    legend([a(1)], [M(1)]);
    subplot(2,1,2)
    a(2) = plot(Data.Uncoupled.Noise_t/1000,Data.Uncoupled.IO_Vs(i,:)*1000,colors(2)); M(2) = 'IO '+string(i)+' Uncoupled';hold on;
    title('IO Response '+string(i)+' Uncoupled'+' '+char(Simulation))
    legend([a(2)], [M(2)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
end
if Params.save == "True"
    for i=1:colIO
        saveas(h(i), fullfile(Params.Both.fname,sprintf('IO_Responses_Both_%d',i)), 'jpeg');
    end
end





start = int64(Params.window(1)/Params.dtime);%-Data.Coupled.Noise_t(1));
finish = int64(Params.window(2)/Params.dtime);%-Data.Coupled.Noise_t(1));
for i = 1:colIO
    h(i) = figure('Renderer', 'painters', 'visible', Params.show); hold on;
    a(1) = plot(Data.Coupled.Noise_t(start:finish)/1000,Data.Coupled.IO_Vs(i,(start:finish))*1000); M(1) = 'IO '+string(i)+' Coupled';
    a(2) = plot(Data.Uncoupled.Noise_t(start:finish)/1000,Data.Uncoupled.IO_Vs(i,(start:finish))*1000); M(2) = 'IO '+string(i)+' Uncoupled';
    title(string(string('IO Cell Responses 1')+' '+char(Simulation)))
    legend([a(:)], [M(:)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000 Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
end

if Params.save == "True"
    for i=1:colIO
        saveas(h(i), fullfile(Params.Both.fname,sprintf('IO_Responses_Window_%d',i)), 'jpeg');
    end
end

colors = ['k','r'];
for i = 1:colIO
    h(i) = figure('Renderer', 'painters', 'visible', Params.show); hold on;
    subplot(2,1,1)
    a(1) = plot(Data.Coupled.Noise_t(start:finish)/1000,Data.Coupled.IO_Vs(i,(start:finish))*1000,colors(1)); M(1) = 'IO '+string(i)+' Coupled';hold on;
    legend([a(1)], [M(1)]);
    title('IO Response Window '+string(i)+' Coupled'+' '+char(Simulation))
    subplot(2,1,2)
    a(2) = plot(Data.Uncoupled.Noise_t(start:finish)/1000,Data.Uncoupled.IO_Vs(i,(start:finish))*1000,colors(2)); M(2) = 'IO '+string(i)+' Uncoupled';hold on;
    title('IO Response Window '+string(i)+' Uncoupled'+' '+char(Simulation))
    legend([a(2)], [M(2)]);
    xlabel('t (seconds)')
    ylabel('V (mV)')
end
if Params.save == "True"
    for i=1:colIO
        saveas(h(i), fullfile(Params.Both.fname,sprintf('IO_Responses_Both_Window_%d',i)), 'jpeg');
    end
end

show = Params.show;
save = Params.save;

f1 = waitbar(1/2,'IO Plots Finished');close(f)


noisee = 0:0.025:120000;
[~,collDCNcoup] = size(Data.Coupled.DCN_firingrate);
[~,collPCcoup] = size(Data.Coupled.PC_firingrate);
[~,collIOcoup] = size(Data.Coupled.IO_firingrate);
[~,collDCNunc] = size(Data.Uncoupled.DCN_firingrate);
[~,collPCunc] = size(Data.Uncoupled.PC_firingrate);
[~,collIOunc] = size(Data.Uncoupled.IO_firingrate);
h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M1 = 'PC Coupled'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC Uncoupled'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('PC Firing Rates Coupled/Uncoupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M1 = 'PC Coupled'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC Uncoupled'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000 Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('PC Firing Rates Coupled/Uncoupled Window')+' '+char(Simulation));
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'PC_FiringRates'), 'jpeg');
end


h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M1 = 'DCN Coupled'; hold on;
a2 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M2 = 'DCN Uncoupled'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN Firing Rates Coupled/Uncoupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M1 = 'DCN Coupled'; hold on;
a2 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M2 = 'DCN Uncoupled'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN Firing Rates Coupled/Uncoupled Window')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'DCN_FiringRates'), 'jpeg');
end


h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate/1000); hold on;
M1 = 'IO Coupled'; hold on;
a2 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate/1000); hold on;
M2 = 'IO Uncoupled'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO Firing Rates Coupled/Uncoupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate/1000); hold on;
M1 = 'IO Coupled'; hold on;
a2 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate/1000); hold on;
M2 = 'IO Uncoupled'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO Firing Rates Coupled/Uncoupled Window')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'IO_FiringRates'), 'jpeg');
end










h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN PC Firing Rates Coupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN PC Firing Rates Uncoupled')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'DCN_PC_FiringRates'), 'jpeg');
end

h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN PC Firing Rates Coupled Window')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M1 = 'DCN'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]);
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('DCN PC Firing Rates Uncoupled Window')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'DCN_PC_FiringRates_Window'), 'jpeg');
end




h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO PC Firing Rates Coupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO PC Firing Rates Uncoupled')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'IO_PC_FiringRates'), 'jpeg');
end

h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collPCcoup)/1000,Data.Coupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]); 
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO PC Firing Rates Coupled Window')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collPCunc)/1000,Data.Uncoupled.PC_firingrate); hold on;
M2 = 'PC'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]); 
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO PC Firing Rates Uncoupled Window')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'IO_PC_FiringRates_Window'), 'jpeg');
end












h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO DCN Firing Rates Coupled')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO DCN Firing Rates Uncoupled')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'IO_DCN_FiringRates'), 'jpeg');
end

h = figure('Renderer', 'painters', 'visible', show);
subplot(2,1,1)
a1 = plot(noisee(1:collIOcoup)/1000,Data.Coupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collDCNcoup)/1000,Data.Coupled.DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]); 
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO DCN Firing Rates Coupled Window')+' '+char(Simulation))
subplot(2,1,2)
a1 = plot(noisee(1:collIOunc)/1000,Data.Uncoupled.IO_firingrate); hold on;
M1 = 'IO'; hold on;
a2 = plot(noisee(1:collDCNunc)/1000,Data.Uncoupled.DCN_firingrate); hold on;
M2 = 'DCN'; hold on;
xlim([Params.window(1)/1000+Data.Coupled.Noise_t(1)/1000  Params.window(2)/1000+Data.Coupled.Noise_t(1)/1000]); 
legend(M1, M2);
xlabel('t (seconds)')
ylabel('Rate (Hz)')
title(string('IO DCN Firing Rates Uncoupled Window')+' '+char(Simulation))
legend(M1, M2);
if save == "True"
    saveas(h, fullfile(Params.Both.fname,'IO_DCN_FiringRates_Window'), 'jpeg');
end

f2 = waitbar(2/2,'Firing Rate Plots Finished');close(f1);pause(.5);close(f2)
f3 = waitbar(2/2,'Coupled/Uncoupled Plots Finished');pause(2);close(f3)


end