function [Params] = params(save,show,window,time,time_Cov,Coupled_mat,Uncoupled_mat)
datetime.setDefaultFormats('default','HH_mm');
Params = struct;
Params.time = string(datetime);
Params.save = save; %"True"; %"False"
Params.show = show; %'off';%'off'
Params.window = window; %[1, 3000];
Params.timepreceding = {time/0.025;time/0.025}; %{100/0.025;100/0.025};
Params.timepreceding_Cov = {time_Cov/0.025;time_Cov/0.025}; %{100/0.025;100/0.025};
Params.Both.fname = char("Figures\"+date+"\"+Params.time+"\No Plasticity\"+"CoupledUncoupled");
if Params.save == "True"
    mkdir(Params.Both.fname);   %create the directory
end
%% Coupled Parameters
Params.Coupled = struct;
Params.Coupled.mat = 'Data\'+string(Coupled_mat)+'.mat'; %'Data\mydata_coupled.mat';
Params.Coupled.simul = "Coupled";
Params.Coupled.fname = char("Figures\"+date+"\"+Params.time+"\No Plasticity\"+string(Params.Coupled.simul));
if Params.save == "True"
    mkdir(Params.Coupled.fname);   %create the directory
end
%% Uncoupled Parameters
Params.Uncoupled = struct;
Params.Uncoupled.mat = 'Data\'+string(Uncoupled_mat)+'.mat'; %'Data\mydata_uncoupled.mat';
Params.Uncoupled.simul = "Uncoupled";
Params.Uncoupled.fname = char("Figures\"+date+"\"+Params.time+"\No Plasticity\"+string(Params.Uncoupled.simul));
if Params.save == "True"
    mkdir(Params.Uncoupled.fname);   %create the directory
end
end

