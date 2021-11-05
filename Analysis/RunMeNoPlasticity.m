clear all;close all;clc
%% Add Path
addpath('Data','Functions')
%% Set Parameters & Load Data
datetime.setDefaultFormats('default','HH_mm');
date = string(datetime);
save('Date.mat', 'date');
load('Date.mat');
Params = struct;
Params.time = date;
Show = 'on';Save = "True";
Simulation = 'Before STDP';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% No Plasticity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Params,Data_beforeSTDP] = StartUp(Save,Show,[1000 1500],1,300,300,'MyData_Coupled_NoPlasticity_1','MyData_Uncoupled_NoPlasticity_1',"No Plasticity",Params); 
%% Choose Plots
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
SimplePlots("Coupled",Simulation,Data_beforeSTDP,Params);
%%
ComplexPlots("Coupled",Simulation,Data_beforeSTDP,Params);
%% Uncoupled
SimplePlots("Uncoupled",Simulation,Data_beforeSTDP,Params);
%%
ComplexPlots("Uncoupled",Simulation,Data_beforeSTDP,Params);
%% Coupled-Uncoupled
CoupUncoup(Data_beforeSTDP,Params,Simulation)
%% Covariance Plots
% Coupled
CovPlots("Coupled",Simulation,Data_beforeSTDP,Params,1) % 1
CovPlotsNoise("Coupled",Simulation,Data_beforeSTDP,Params,1) % 1
CovPlots("Uncoupled",Simulation,Data_beforeSTDP,Params,1) % 1
CovPlotsNoise("Uncoupled",Simulation,Data_beforeSTDP,Params,1) % 1
%% Statistics%% Uncoupled

% Statistic = struct;
% [Statistic] = Statistics("Coupled",Data_beforeSTDP,Params,Statistic);
% [Statistic] = Statistics("Uncoupled",Data_beforeSTDP,Params,Statistic);
% %%
% currDate = strrep(datestr(datetime), ':', '_');
% mkdir('Statistics',currDate)
% save(fullfile('Statistics',currDate,'Statistic_NoPlasticity.mat'));



