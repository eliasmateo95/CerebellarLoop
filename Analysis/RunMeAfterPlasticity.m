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
Simulation = 'After STDP';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% After Plasticity %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Params,Data_forAfterSTDP] = StartUp(Save,Show,[1000 1500],1,300,300,'MyData_Coupled_Plasticity_0','MyData_Uncoupled_Plasticity_0',"After Plasticity",Params); 
[Data_AfterSTDP] = AfterSTDP(Data_forAfterSTDP,Params);
% Choose Plots Coupled
NOISE = 'n'; PC_NOISE = 'n'; ALL_NOISE = 'n'; 
IO_DCN_PC = 'y'; PC = 'n'; DCN = 'n'; IO = 'n'; 
CS_TRIG_NOISE = 'n'; CS_TRIG_SS = 'n'; CS_TRIG_DCN = 'n'; 
DCN_TRIG_NOISE = 'n'; 
FIRING = 'n'; CORR_FIRING = 'n'; 
RASTER = 'n'; 
NOISE_COV = 'n'; PC_COV = 'n'; 
WATERFALL_NOISE = 'n'; WATERFALL_IO_VS = 'n';
[Params] = PlotChoice(Params,"Coupled",NOISE,PC_NOISE,ALL_NOISE,IO_DCN_PC,PC,DCN,IO,CS_TRIG_NOISE,CS_TRIG_SS,CS_TRIG_DCN,DCN_TRIG_NOISE,FIRING,CORR_FIRING,RASTER,NOISE_COV,PC_COV,WATERFALL_NOISE,WATERFALL_IO_VS);
[Params] = PlotChoice(Params,"Uncoupled",NOISE,PC_NOISE,ALL_NOISE,IO_DCN_PC,PC,DCN,IO,CS_TRIG_NOISE,CS_TRIG_SS,CS_TRIG_DCN,DCN_TRIG_NOISE,FIRING,CORR_FIRING,RASTER,NOISE_COV,PC_COV,WATERFALL_NOISE,WATERFALL_IO_VS);
clear NOISE PC_NOISE ALL_NOISE IO_DCN_PC PC DCN IO CS_TRIG_NOISE CS_TRIG_SS CS_TRIG_DCN DCN_TRIG_NOISE FIRING CORR_FIRING RASTER NOISE_COV PC_COV WATERFALL_NOISE WATERFALL_IO_VS
%% Coupled 
SimplePlotsAfterSTDP("Coupled",Simulation,Data_AfterSTDP,Params);
%%
ComplexPlotsAfterSTDP("Coupled",Simulation,Data_AfterSTDP,Params);
%% Uncoupled 
SimplePlotsAfterSTDP("Uncoupled",Simulation,Data_AfterSTDP,Params);
ComplexPlotsAfterSTDP("Uncoupled",Simulation,Data_AfterSTDP,Params);
%% Coupled-Uncoupled
CoupUncoup(Data_AfterSTDP,Params,Simulation)
%% Covariance Plots
% Coupled
CovPlots("Coupled",Simulation,Data_AfterSTDP,Params,1) % 1
CovPlots("Coupled",Simulation,Data_AfterSTDP,Params,2) % 2
CovPlotsNoise("Coupled",Simulation,Data_AfterSTDP,Params,1) % 1
CovPlotsNoise("Coupled",Simulation,Data_AfterSTDP,Params,2) % 2
% Uncoupled
CovPlots("Uncoupled",Simulation,Data_AfterSTDP,Params,1) % 1
CovPlots("Uncoupled",Simulation,Data_AfterSTDP,Params,2) % 2
%%
CovPlotsNoise("Uncoupled",Simulation,Data_AfterSTDP,Params,1) % 1
%%
CovPlotsNoise("Uncoupled",Simulation,Data_AfterSTDP,Params,2) % 2
%% Statistics
Statistic = struct;
[Statistic] = Statistics("Coupled",Data_AfterSTDP,Params,Statistic);
[Statistic] = Statistics("Uncoupled",Data_AfterSTDP,Params,Statistic);
save('Statistic_AfterPlasticity.mat', '-struct', 'Statistic')