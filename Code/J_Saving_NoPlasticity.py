from H_Synapses_NoPlasticity import *

###########################################################################################################################
###########################################################################################################################
##################################################### COUPLED SCENARIO ####################################################
###########################################################################################################################
###########################################################################################################################
Output_Coupled = Struct()
Output_Coupled.Noise = Struct()
Output_Coupled.PC = Struct()
Output_Coupled.IO = Struct()
Output_Coupled.DCN = Struct()
Output_Coupled.Noise.time = Noise_statemon.t/ms
Output_Coupled.Noise.I = [Noise_statemon.I]
#####################################################################
########################### PURKINJE CELLS ##########################
#####################################################################
PC_spikes = []
for i in range(0,PC_Spikemon_Coupled_noSTDP.values('t').__len__(),1):
    PC_spikes.append(PC_Spikemon_Coupled_noSTDP.values('t')[i])
Output_Coupled.PC.V = PC_Statemon_Coupled_noSTDP.v
#         del PC_Statemon_Coupled_noSTDP.v
Output_Coupled.PC.spikes = PC_spikes
del PC_spikes
Output_Coupled.PC.noise = PC_Statemon_Coupled_noSTDP.I_Noise
#         del PC_Statemon_Coupled_noSTDP.I_Noise
Output_Coupled.PC.firingratesmooth = PC_rate_Coupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del PC_rate_Coupled_noSTDP
#####################################################################
###################### INFERIOR OLIVARY CELLS #######################
#####################################################################
IO_spikes = []
for i in range(0,IO_Spikemon_Coupled_noSTDP.values('t').__len__(),1):
    IO_spikes.append(IO_Spikemon_Coupled_noSTDP.values('t')[i])
Output_Coupled.IO.Vs = IO_Statemon_Coupled_noSTDP.Vs
#         del IO_Statemon_Coupled_noSTDP.Vs
Output_Coupled.IO.spikes = IO_spikes
del IO_spikes
Output_Coupled.IO.firingratesmooth = IO_rate_Coupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del IO_rate_Coupled_noSTDP
#####################################################################
################### DEEP CEREBELLAR NUCLEI CELLS ####################
#####################################################################
DCN_spikes = []
for i in range(0,DCN_Spikemon_Coupled_noSTDP.values('t').__len__(),1):
    DCN_spikes.append(DCN_Spikemon_Coupled_noSTDP.values('t')[i])
Output_Coupled.DCN.v = DCN_Statemon_Coupled_noSTDP.v
#         del DCN_Statemon_Coupled_noSTDP.v
Output_Coupled.DCN.spikes = DCN_spikes
del DCN_spikes
Output_Coupled.DCN.firingratesmooth = DCN_rate_Coupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del DCN_rate_Coupled_noSTDP
#####################################################################
############################## SAVING ###############################
#####################################################################
import numpy, scipy.io
local = datetime.datetime.now()
scipy.io.savemat('MyData_Coupled_NoPlasticity_'+str(noise_seed)+'.mat', mdict={'Output_Coupled': Output_Coupled})  
###########################################################################################################################
###########################################################################################################################
################################################### UNCOUPLED SCENARIO ####################################################
###########################################################################################################################
###########################################################################################################################
Output_Uncoupled = Struct()
Output_Uncoupled.Noise = Struct()
Output_Uncoupled.PC = Struct()
Output_Uncoupled.IO = Struct()
Output_Uncoupled.DCN = Struct()
Output_Uncoupled.Noise.time = Noise_statemon.t/ms
Output_Uncoupled.Noise.I = [Noise_statemon.I]
#####################################################################
########################### PURKINJE CELLS ##########################
#####################################################################
PC_spikes = []
for i in range(0,PC_Spikemon_Uncoupled_noSTDP.values('t').__len__(),1):
    PC_spikes.append(PC_Spikemon_Uncoupled_noSTDP.values('t')[i])
Output_Uncoupled.PC.V = PC_Statemon_Uncoupled_noSTDP.v
#         del PC_Statemon_Uncoupled_noSTDP.v
Output_Uncoupled.PC.spikes = PC_spikes
del PC_spikes
Output_Uncoupled.PC.noise = PC_Statemon_Uncoupled_noSTDP.I_Noise
#         del PC_Statemon_Uncoupled_noSTDP.I_Noise
Output_Uncoupled.PC.firingratesmooth = PC_rate_Uncoupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del PC_rate_Uncoupled_noSTDP
#####################################################################
###################### INFERIOR OLIVARY CELLS #######################
#####################################################################
IO_spikes = []
for i in range(0,IO_Spikemon_Uncoupled_noSTDP.values('t').__len__(),1):
    IO_spikes.append(IO_Spikemon_Uncoupled_noSTDP.values('t')[i])
Output_Uncoupled.IO.Vs = IO_Statemon_Uncoupled_noSTDP.Vs
#         del IO_Statemon_Uncoupled_noSTDP.Vs
Output_Uncoupled.IO.spikes = IO_spikes
del IO_spikes
Output_Uncoupled.IO.firingratesmooth = IO_rate_Uncoupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del IO_rate_Uncoupled_noSTDP
#####################################################################
################### DEEP CEREBELLAR NUCLEI CELLS ####################
#####################################################################
DCN_spikes = []
for i in range(0,DCN_Spikemon_Uncoupled_noSTDP.values('t').__len__(),1):
    DCN_spikes.append(DCN_Spikemon_Uncoupled_noSTDP.values('t')[i])
Output_Uncoupled.DCN.v = DCN_Statemon_Uncoupled_noSTDP.v
#         del DCN_Statemon_Uncoupled_noSTDP.v
Output_Uncoupled.DCN.spikes = DCN_spikes
del DCN_spikes
Output_Uncoupled.DCN.firingratesmooth = DCN_rate_Uncoupled_noSTDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del DCN_rate_Uncoupled_noSTDP
#####################################################################
############################## SAVING ###############################
#####################################################################
import numpy, scipy.io
local = datetime.datetime.now()
scipy.io.savemat('MyData_Uncoupled_NoPlasticity_'+str(noise_seed)+'.mat', mdict={'Output_Uncoupled': Output_Uncoupled}) 
