%[Params] = params(SAVE,SHOW,WINDOW,TIME,TIME_COV,COUPLED_MAT,UNCOUPLED_MAT);
% SAVE = "True"/"False" to save figures to folders
% SHOW = 'on'/'off' to show plots
% WINDOW = [start end] in ms for extra plots
% TIME (ms) = is time preceding the complex spikes
% TIME_COV = TIME but for the covariance matrix due to heavy picture
% COUPLED/UNCOUPLED_MAT = names OF files for coupled/uncoupled situation (without the .mat). 
function [Params,Data] = StartUp(save,show,window,dtime,time,time_Cov,Coupled_mat,Uncoupled_mat,folder,Params)
[Params] = params(save,show,window,dtime,time,time_Cov,Coupled_mat,Uncoupled_mat,folder,Params);
[Data] = data(Params);
end


function [Params] = params(save,show,window,dtime,time,time_Cov,Coupled_mat,Uncoupled_mat,folder,Params)
Params.save = save; %"True"; %"False"
Params.show = show; %'off';%'off'
Params.window = window; %[1, 3000];
Params.folder = folder;
Params.dtime = dtime;
Params.timepreceding = {time/dtime;time/dtime}; %{100/0.025;100/0.025};
Params.timepreceding_Cov = {time_Cov/dtime;time_Cov/dtime}; %{100/0.025;100/0.025};
Params.Both.fname = char("Figures\"+date+"\"+Params.time+"\"+Params.folder+"\"+"CoupledUncoupled");
if Params.save == "True"
    mkdir(Params.Both.fname);   %create the directory
end
%% Coupled Parameters
Params.Coupled = struct;
Params.Coupled.mat = 'Data\'+string(Coupled_mat)+'.mat'; %'Data\mydata_coupled.mat';
Params.Coupled.simul = "Coupled";
Params.Coupled.fname = char("Figures\"+date+"\"+Params.time+"\"+Params.folder+"\"+string(Params.Coupled.simul));
if Params.save == "True"
    mkdir(Params.Coupled.fname);   %create the directory
end
%% Uncoupled Parameters
Params.Uncoupled = struct;
Params.Uncoupled.mat = 'Data\'+string(Uncoupled_mat)+'.mat'; %'Data\mydata_uncoupled.mat';
Params.Uncoupled.simul = "Uncoupled";
Params.Uncoupled.fname = char("Figures\"+date+"\"+Params.time+"\"+Params.folder+"\"+string(Params.Uncoupled.simul));
if Params.save == "True"
    mkdir(Params.Uncoupled.fname);   %create the directory
end
end

function [Data] = data(Params)
    Data = struct;
    if isfield(Params, 'Coupled' )
        load(Params.Coupled.mat);
        Data.Coupled = struct;
        Data.Coupled.Noise_t = Output_Coupled.Noise.time;
        Data.Coupled.Noise_I = Output_Coupled.Noise.I;
        Data.Coupled.PC_v = Output_Coupled.PC.V;
        Data.Coupled.PC_spikes = Output_Coupled.PC.spikes;
        Data.Coupled.PC_noise = Output_Coupled.PC.noise;
        Data.Coupled.PC_firingrate = Output_Coupled.PC.firingratesmooth;
        Data.Coupled.IO_Vs = Output_Coupled.IO.Vs;
        Data.Coupled.IO_spikes = Output_Coupled.IO.spikes;
        Data.Coupled.IO_firingrate = Output_Coupled.IO.firingratesmooth;
        Data.Coupled.DCN_v = Output_Coupled.DCN.v;
        Data.Coupled.DCN_spikes = Output_Coupled.DCN.spikes;
        Data.Coupled.DCN_firingrate = Output_Coupled.DCN.firingratesmooth;        
    end
    
    if isfield(Params, 'Uncoupled' )    
        load(Params.Uncoupled.mat);
        Data.Uncoupled = struct;
        Data.Uncoupled.Noise_t = Output_Uncoupled.Noise.time;
        Data.Uncoupled.Noise_I = Output_Uncoupled.Noise.I;
        Data.Uncoupled.PC_v = Output_Uncoupled.PC.V;
        Data.Uncoupled.PC_spikes = Output_Uncoupled.PC.spikes;
        Data.Uncoupled.PC_noise = Output_Uncoupled.PC.noise;
        Data.Uncoupled.PC_firingrate = Output_Uncoupled.PC.firingratesmooth;
        Data.Uncoupled.IO_Vs = Output_Uncoupled.IO.Vs;
        Data.Uncoupled.IO_spikes = Output_Uncoupled.IO.spikes;
        Data.Uncoupled.IO_firingrate = Output_Uncoupled.IO.firingratesmooth;
        Data.Uncoupled.DCN_v = Output_Uncoupled.DCN.v;
        Data.Uncoupled.DCN_spikes = Output_Uncoupled.DCN.spikes;
        Data.Uncoupled.DCN_firingrate = Output_Uncoupled.DCN.firingratesmooth;
    end
end