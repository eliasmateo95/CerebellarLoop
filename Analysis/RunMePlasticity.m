clear all;close all;clc
%% Add Path
addpath('Data','Functions')
%% Set Parameters & Load Data
% datetime.setDefaultFormats('default','HH_mm');
% date = string(datetime);
% save('Date.mat', 'date') 
load('Date.mat');
Params = struct;
Params.time = date;
Show = 'off';
Save = "True";
Simulation = 'STDP';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plasticity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Params,Data_forSTDP] = StartUp(Save,Show,[1 5000],600,300,'mydata_coupled_Plasticity','mydata_uncoupled_Plasticity',"Plasticity",Params); 
[Data_STDP] = STDP(Data_forSTDP,Params);
%% Choose Plots Coupled
NOISE = 'y'; PC_NOISE = 'y'; ALL_NOISE = 'y'; 
IO_DCN_PC = 'y'; PC = 'y'; DCN = 'y'; IO = 'y'; 
CS_TRIG_NOISE = 'y'; CS_TRIG_SS = 'y'; CS_TRIG_DCN = 'y'; 
DCN_TRIG_NOISE = 'y'; 
FIRING = 'y'; CORR_FIRING = 'y'; 
RASTER = 'y'; 
NOISE_COV = 'y'; PC_COV = 'y'; 
WATERFALL_NOISE = 'y'; WATERFALL_IO_VS = 'y';
[Params] = PlotChoice(Params,"Coupled",NOISE,PC_NOISE,ALL_NOISE,IO_DCN_PC,PC,DCN,IO,CS_TRIG_NOISE,CS_TRIG_SS,CS_TRIG_DCN,DCN_TRIG_NOISE,FIRING,CORR_FIRING,RASTER,NOISE_COV,PC_COV,WATERFALL_NOISE,WATERFALL_IO_VS);
[Params] = PlotChoice(Params,"Uncoupled",NOISE,PC_NOISE,ALL_NOISE,IO_DCN_PC,PC,DCN,IO,CS_TRIG_NOISE,CS_TRIG_SS,CS_TRIG_DCN,DCN_TRIG_NOISE,FIRING,CORR_FIRING,RASTER,NOISE_COV,PC_COV,WATERFALL_NOISE,WATERFALL_IO_VS);
clear NOISE PC_NOISE ALL_NOISE IO_DCN_PC PC DCN IO CS_TRIG_NOISE CS_TRIG_SS CS_TRIG_DCN DCN_TRIG_NOISE FIRING CORR_FIRING RASTER NOISE_COV PC_COV WATERFALL_NOISE WATERFALL_IO_VS
%% Coupled 
SimplePlots("Coupled",Simulation,Data_STDP,Params);
ComplexPlots("Coupled",Simulation,Data_STDP,Params);
%% Uncoupled
SimplePlots("Uncoupled",Simulation,Data_STDP,Params);
ComplexPlots("Uncoupled",Simulation,Data_STDP,Params);
%% Coupled-Uncoupled
CoupUncoup(Data_STDP,Params,Simulation)
%% Covariance Plots
% Coupled
CovPlots("Coupled",Simulation,Data_STDP,Params,1) % 1
CovPlots("Coupled",Simulation,Data_STDP,Params,2) % 2
CovPlotsNoise("Coupled",Simulation,Data_STDP,Params,1) % 1
CovPlotsNoise("Coupled",Simulation,Data_STDP,Params,2) % 2
% Uncoupled
CovPlots("Uncoupled",Simulation,Data_STDP,Params,1) % 1
CovPlots("Uncoupled",Simulation,Data_STDP,Params,2) % 2
CovPlotsNoise("Uncoupled",Simulation,Data_STDP,Params,1) % 1
%%
CovPlotsNoise("Uncoupled",Simulation,Data_STDP,Params,2) % 2
%% Statistics
Statistic = struct;
[Statistic] = Statistics("Coupled",Data_STDP,Params,Statistic);
[Statistic] = Statistics("Uncoupled",Data_STDP,Params,Statistic);
save('Statistic_Plasticity.mat', '-struct', 'Statistic')