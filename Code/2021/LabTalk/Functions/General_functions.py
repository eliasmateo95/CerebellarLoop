from brian2 import *
from brian2tools import *
import brian2.numpy_ as np
import datetime
import pickle
import random
from IPython.display import display
import brian2.numpy_ as np # the numpy that comes bundled with it
from ipywidgets import interact, interactive # for some neat interactions
import ipywidgets as widgets
import matplotlib.pyplot as plt # for neat plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
import matplotlib.gridspec as gridspec
from random import randrange
import os
import sys 
import datetime
import scipy.io as sio
from scipy import signal
import pandas as pd
from statistics import variance
import seaborn as sns



class Struct:
    pass

def visualise(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

def rand_params(Parameter,Unit,N_Cells,Step):
    if N_Cells == 1:
        Param_vector = [Parameter*Unit]
    else:
        Nn = [int(N_Cells/2), N_Cells-int(N_Cells/2)] 
        shuffle(Nn)
        Base = int(1/Step)
        Start = int(Base*Parameter)
        Begin = Start - Nn[0]
        End = Start + Nn[1]
        Param_vector = [x / float(Base) for x in range(Begin, End, 1)]*Unit
        shuffle(Param_vector)
    return Param_vector


def cells_connected_to_noise(PC_DCN_Synapse_Targets,PC_DCN_Synapse_Sources,DCN_IO_Synapse_Targets,IO_PC_Synapse_Sources):
    IO_Cells_Connected = []
    for ii in range(0,size(PC_DCN_Synapse_Targets)):
        IO_Cells_Connected.append(DCN_IO_Synapse_Targets[PC_DCN_Synapse_Targets[ii]])
    DCN_Cells_Connected = PC_DCN_Synapse_Targets

    IO_Cell_to_show = []
    PC_Cell_to_show = []
    for ii in range(0,size(IO_PC_Synapse_Sources)):
        if IO_Cells_Connected[ii] == IO_PC_Synapse_Sources[ii]:
            IO_Cell_to_show.append(IO_Cells_Connected[ii])
            PC_Cell_to_show.append(PC_DCN_Synapse_Sources[ii])
            
    return IO_Cell_to_show,PC_Cell_to_show,DCN_Cells_Connected,IO_Cells_Connected

def most_frequent(List):
    counter = 0
    num = List[0]      
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

# def Create_output(Name,width,Noise_statemon,Noise_extended_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Noise_statemon_Coupled,Noise_extended_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled):
    
#     Output = {}

#     Output["Noise"] = {}
#     Output["Noise"]['I'] = Noise_statemon.I
#     Output["Noise"]['weight'] = Noise_statemon.weight
#     Output["Noise"]['time'] = Noise_statemon.t/ms
#     Output["Noise_Extended"] = {}
#     Output["Noise_Extended"]["I"] = Noise_extended_statemon.I
#     Output["Noise_Extended"]["weight"] = Noise_extended_statemon.weight
#     Output["Noise_Extended"]["time"] = Noise_extended_statemon.t/ms
#     Output["PC"] = {}
#     Output["PC"]['v'] = PC_Statemon.v
#     Output["PC"]['w'] = PC_Statemon.w
#     Output["PC"]['I_Noise'] = PC_Statemon.I_Noise
#     Output["PC"]['I_Noise_empty'] = PC_Statemon.I_Noise_empty
#     Output["PC"]['I_intrinsic'] = PC_Statemon.I_intrinsic
#     Output["PC"]['tauw'] = PC_Statemon.tauw
#     Output["PC"]['recent_rate'] = PC_Statemon.recent_rate
#     Output["PC"]['New_recent_rate'] = PC_Statemon.New_recent_rate
#     Output["PC"]['Spikemon'] = PC_Spikemon.t/ms
#     PC_Spikemon_Cells = [[]]*PC_Spikemon.values('t').__len__()
#     for ii in range(0,PC_Spikemon.values('t').__len__()):
#         PC_Spikemon_Cells[ii] = PC_Spikemon.values('t')[ii]
#     Output["PC"]['Spikemon_Cells'] = PC_Spikemon_Cells
#     Output["PC"]['Rate'] = PC_rate.smooth_rate(window='gaussian', width=width)/Hz
#     Output["PC"]['Rate_time'] = PC_rate.t/ms
#     Output["DCN"] = {}
#     Output["DCN"]['v'] = DCN_Statemon.v
#     Output["DCN"]['I_PC'] = DCN_Statemon.I_PC
#     Output["DCN"]['w'] = DCN_Statemon.w
#     Output["DCN"]['Spikemon'] = DCN_Spikemon.t/ms
#     Output["DCN"]['Rate'] = DCN_rate.smooth_rate(window='gaussian', width=width)/Hz
#     Output["IO"] = {}
#     Output["IO"]['Vs'] = IO_Statemon.Vs
#     Output["IO"]['Vd'] = IO_Statemon.Vd
#     Output["IO"]['Va'] = IO_Statemon.Va
#     Output["IO"]['I_c'] = IO_Statemon.I_c
#     Output["IO"]['Iapp_s'] = IO_Statemon.Iapp_s
#     Output["IO"]['Iapp_d'] = IO_Statemon.Iapp_d
#     Output["IO"]['I_IO_DCN'] = IO_Statemon.I_IO_DCN
#     Output["IO"]['Spikemon'] = IO_Spikemon.t/ms
#     IO_Spikemon_Cells = [[]]*IO_Spikemon.values('t').__len__()
#     for ii in range(0,IO_Spikemon.values('t').__len__()):
#         IO_Spikemon_Cells[ii] = IO_Spikemon.values('t')[ii]
#     Output["IO"]['Spikemon_Cells'] = IO_Spikemon_Cells
#     Output["IO"]['Rate'] = IO_rate.smooth_rate(window='gaussian', width=width)/Hz
#     if mon_N_PC:
#         Output["mon_N_PC"] = {}
#         if Name == "STDP":
#             Output["mon_N_PC"]['a_PC'] = mon_N_PC.a_PC
#             Output["mon_N_PC"]['a_IO'] = mon_N_PC.a_IO
#             Output["mon_N_PC"]['I'] = mon_N_PC.I
#             Output["mon_N_PC"]['delta_weight'] = mon_N_PC.delta_weight
#         else:
#             Output["mon_N_PC"]['rho_PF'] = mon_N_PC.rho_PF
#             Output["mon_N_PC"]['rho_PC'] = mon_N_PC.rho_PC
#             Output["mon_N_PC"]['phi'] = mon_N_PC.phi
#             Output["mon_N_PC"]['thresh_M'] = mon_N_PC.thresh_M
#             Output["mon_N_PC"]['delta_weight'] = mon_N_PC.delta_weight
#             Output["mon_N_PC"]['new_weight'] = mon_N_PC.new_weight


#     Output["Noise_Coupled"] = {}
#     Output["Noise_Coupled"]['I'] = Noise_statemon_Coupled.I
#     Output["Noise_Coupled"]['weight'] = Noise_statemon_Coupled.weight
#     Output["Noise_Coupled"]['time'] = Noise_statemon.t/ms
#     Output["Noise_Extended_Coupled"] = {}
#     Output["Noise_Extended_Coupled"]["I"] = Noise_extended_statemon_Coupled.I
#     Output["Noise_Extended_Coupled"]["weight"] = Noise_extended_statemon_Coupled.weight
#     Output["Noise_Extended_Coupled"]["time"] = Noise_extended_statemon.t/ms
#     Output["PC_Coupled"] = {}
#     Output["PC_Coupled"]['v'] = PC_Statemon_Coupled.v
#     Output["PC_Coupled"]['w'] = PC_Statemon_Coupled.w
#     Output["PC_Coupled"]['I_Noise'] = PC_Statemon_Coupled.I_Noise
#     Output["PC_Coupled"]['I_Noise_empty'] = PC_Statemon_Coupled.I_Noise_empty
#     Output["PC_Coupled"]['I_intrinsic'] = PC_Statemon_Coupled.I_intrinsic
#     Output["PC_Coupled"]['tauw'] = PC_Statemon_Coupled.tauw
#     Output["PC_Coupled"]['recent_rate'] = PC_Statemon_Coupled.recent_rate
#     Output["PC_Coupled"]['New_recent_rate'] = PC_Statemon_Coupled.New_recent_rate
#     Output["PC_Coupled"]['Spikemon'] = PC_Spikemon_Coupled.t/ms
#     PC_Spikemon_Cells_Coupled = [[]]*PC_Spikemon_Coupled.values('t').__len__()
#     for ii in range(0,PC_Spikemon_Coupled.values('t').__len__()):
#         PC_Spikemon_Cells_Coupled[ii] = PC_Spikemon_Coupled.values('t')[ii]
#     Output["PC_Coupled"]['Spikemon_Cells'] = PC_Spikemon_Cells_Coupled
#     Output["PC_Coupled"]['Rate'] = PC_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz
#     Output["PC_Coupled"]['Rate_time'] = PC_rate_Coupled.t/ms
#     Output["DCN_Coupled"] = {}
#     Output["DCN_Coupled"]['v'] = DCN_Statemon_Coupled.v
#     Output["DCN_Coupled"]['I_PC'] = DCN_Statemon_Coupled.I_PC
#     Output["DCN_Coupled"]['w'] = DCN_Statemon_Coupled.w
#     Output["DCN_Coupled"]['Spikemon'] = DCN_Spikemon_Coupled.t/ms
#     Output["DCN_Coupled"]['Rate'] = DCN_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz
#     Output["IO_Coupled"] = {}
#     Output["IO_Coupled"]['Vs'] = IO_Statemon_Coupled.Vs
#     Output["IO_Coupled"]['Vd'] = IO_Statemon_Coupled.Vd
#     Output["IO_Coupled"]['Va'] = IO_Statemon_Coupled.Va
#     Output["IO_Coupled"]['I_c'] = IO_Statemon_Coupled.I_c
#     Output["IO_Coupled"]['Iapp_s'] = IO_Statemon_Coupled.Iapp_s
#     Output["IO_Coupled"]['Iapp_d'] = IO_Statemon_Coupled.Iapp_d
#     Output["IO_Coupled"]['I_IO_DCN'] = IO_Statemon_Coupled.I_IO_DCN
#     Output["IO_Coupled"]['Spikemon'] = IO_Spikemon_Coupled.t/ms
#     IO_Spikemon_Cells_Coupled = [[]]*IO_Spikemon_Coupled.values('t').__len__()
#     for ii in range(0,IO_Spikemon_Coupled.values('t').__len__()):
#         IO_Spikemon_Cells_Coupled[ii] = IO_Spikemon_Coupled.values('t')[ii]
#     Output["IO_Coupled"]['Spikemon_Cells'] = IO_Spikemon_Cells_Coupled
#     Output["IO_Coupled"]['Rate'] = IO_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz
#     if mon_N_PC_Coupled:
#         Output["mon_N_PC_Coupled"] = {}
#         if Name == "STDP":
#             Output["mon_N_PC_Coupled"]['a_PC'] = mon_N_PC.a_PC
#             Output["mon_N_PC_Coupled"]['a_IO'] = mon_N_PC.a_IO
#             Output["mon_N_PC_Coupled"]['I'] = mon_N_PC.I
#             Output["mon_N_PC_Coupled"]['delta_weight'] = mon_N_PC.delta_weight
#         else:
#             Output["mon_N_PC_Coupled"]['rho_PF'] = mon_N_PC_Coupled.rho_PF
#             Output["mon_N_PC_Coupled"]['rho_PC'] = mon_N_PC_Coupled.rho_PC
#             Output["mon_N_PC_Coupled"]['phi'] = mon_N_PC_Coupled.phi
#             Output["mon_N_PC_Coupled"]['thresh_M'] = mon_N_PC_Coupled.thresh_M
#             Output["mon_N_PC_Coupled"]['delta_weight'] = mon_N_PC_Coupled.delta_weight
#             Output["mon_N_PC_Coupled"]['new_weight'] = mon_N_PC_Coupled.new_weight

#     path = os.getcwd()
#     save_path = path+"/Data/"+datetime.datetime.now().strftime("%m-%d")  
#     file_name = 'Output'+str(Name)+"_"+datetime.datetime.now().strftime("%H:%M")+'.mat'
#     completeName = os.path.join(save_path, file_name)

#     sio.savemat(completeName, Output) 



# def Output_func(Name,Output_data,coupled):
#     class Noise:
#         pass
#     class Noise_Extended:
#         pass
#     class PC:
#         pass
#     class DCN:
#         pass
#     class IO:
#         pass
#     class mon_N_PC:
#         pass
    
#     Noise.I = Output_data["Noise"+str(coupled)]['I'].item()
#     Noise.weight = Output_data["Noise"+str(coupled)]['weight'].item()
#     Noise.t = Output_data["Noise"+str(coupled)]['time'].item()

#     Noise_Extended.I = Output_data["Noise_Extended"+str(coupled)]["I"].item()
#     Noise_Extended.weight = Output_data["Noise_Extended"+str(coupled)]["weight"].item()
#     Noise_Extended.time = Output_data["Noise_Extended"+str(coupled)]["time"].item()

#     PC.v = Output_data["PC"+str(coupled)]['v'].item()
#     PC.w = Output_data["PC"+str(coupled)]['w'].item()
#     PC.I_Noise = Output_data["PC"+str(coupled)]['I_Noise'].item()
#     PC.I_Noise_empty = Output_data["PC"+str(coupled)]['I_Noise_empty'].item() 
#     PC.I_intrinsic = Output_data["PC"+str(coupled)]['I_intrinsic'].item()
#     PC.tauw = Output_data["PC"+str(coupled)]['tauw'].item()
#     PC.recent_rate = Output_data["PC"+str(coupled)]['recent_rate'].item()
#     PC.New_recent_rate = Output_data["PC"+str(coupled)]['New_recent_rate'].item()
#     PC.Spikemon = Output_data["PC"+str(coupled)]['Spikemon'].item()
#     PC.rate = Output_data["PC"+str(coupled)]['Rate'].item()
#     PC.rate_time = Output_data["PC"+str(coupled)]['Rate_time'].item()
#     PC.Spikemon_Cells = Output_data["PC"+str(coupled)]['Spikemon_Cells'].item()

#     DCN.v = Output_data["DCN"+str(coupled)]['v'].item()
#     DCN.I_PC = Output_data["DCN"+str(coupled)]['I_PC'].item()
#     DCN.w = Output_data["DCN"+str(coupled)]['w'].item()
#     DCN.Spikemon = Output_data["DCN"+str(coupled)]['Spikemon'].item()
#     DCN.rate = Output_data["DCN"+str(coupled)]['Rate'].item()

#     IO.Vs = Output_data["IO"+str(coupled)]['Vs'].item()
#     IO.Vd = Output_data["IO"+str(coupled)]['Vd'].item()
#     IO.Va = Output_data["IO"+str(coupled)]['Va'].item()
#     IO.I_c = Output_data["IO"+str(coupled)]['I_c'].item()
#     IO.Iapp_s = Output_data["IO"+str(coupled)]['Iapp_s'].item()
#     IO.Iapp_d = Output_data["IO"+str(coupled)]['Iapp_d'].item()
#     IO.I_IO_DCN = Output_data["IO"+str(coupled)]['I_IO_DCN'].item()
#     IO.Spikemon = Output_data["IO"+str(coupled)]['Spikemon'].item()
#     IO.Spikemon_Cells = Output_data["IO"+str(coupled)]['Spikemon_Cells'].item()
#     IO.rate = Output_data["IO"+str(coupled)]['Rate'].item()
    
#     if Name == "STDP":
#         mon_N_PC.a_PC = Output_data["mon_N_PC"+str(coupled)]['a_PC'].item()
#         mon_N_PC.a_IO = Output_data["mon_N_PC"+str(coupled)]['a_IO'].item()
#         mon_N_PC.I = Output_data["mon_N_PC"+str(coupled)]['I'].item()
#         mon_N_PC.delta_weight = Output_data["mon_N_PC"+str(coupled)]['delta_weight'].item()
#     else:
#         mon_N_PC.rho_PF = Output_data["mon_N_PC"+str(coupled)]['rho_PF'].item()
#         mon_N_PC.rho_PC = Output_data["mon_N_PC"+str(coupled)]['rho_PC'].item()
#         mon_N_PC.phi = Output_data["mon_N_PC"+str(coupled)]['phi'].item()
#         mon_N_PC.thresh_M = Output_data["mon_N_PC"+str(coupled)]['thresh_M'].item()
#         mon_N_PC.delta_weight = Output_data["mon_N_PC"+str(coupled)]['delta_weight'].item()
#         mon_N_PC.new_weight = Output_data["mon_N_PC"+str(coupled)]['new_weight'].item()
    
#     return Noise, Noise_Extended, PC, DCN, IO, mon_N_PC