from H_Synapses_Plasticity import *

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
for i in range(0,PC_Spikemon_Coupled_STDP.values('t').__len__(),1):
    PC_spikes.append(PC_Spikemon_Coupled_STDP.values('t')[i])
Output_Coupled.PC.V = PC_Statemon_Coupled_STDP.v
#         del PC_Statemon_Coupled_STDP.v
Output_Coupled.PC.spikes = PC_spikes
del PC_spikes
Output_Coupled.PC.noise = PC_Statemon_Coupled_STDP.I_Noise
#         del PC_Statemon_Coupled_STDP.I_Noise
Output_Coupled.PC.firingratesmooth = PC_rate_Coupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del PC_rate_Coupled_STDP
Output_Coupled.PC.noise_source = mon_N_PC_Coupled.noise_source
#         del mon_N_PC_Coupled.noise_source
Output_Coupled.PC.PC_target = mon_N_PC_Coupled.PC_target
#         del mon_N_PC_Coupled.PC_target
Output_Coupled.PC.weight = mon_N_PC_Coupled.weight
#         del mon_N_PC_Coupled.weight
Output_Coupled.PC.I = mon_N_PC_Coupled.I
#         del mon_N_PC_Coupled.I
Output_Coupled.PC.new_weight = mon_N_PC_Coupled.new_weight
#         del mon_N_PC_Coupled.new_weight
Output_Coupled.PC.delta_weight = mon_N_PC_Coupled.delta_weight
#         del mon_N_PC_Coupled.delta_weight
Output_Coupled.PC.a_PC = mon_N_PC_Coupled.a_PC
#         del mon_N_PC_Coupled.a_PC
Output_Coupled.PC.a_IO = mon_N_PC_Coupled.a_IO
del mon_N_PC_Coupled
#####################################################################
###################### INFERIOR OLIVARY CELLS #######################
#####################################################################
IO_spikes = []
for i in range(0,IO_Spikemon_Coupled_STDP.values('t').__len__(),1):
    IO_spikes.append(IO_Spikemon_Coupled_STDP.values('t')[i])
Output_Coupled.IO.Vs = IO_Statemon_Coupled_STDP.Vs
#         del IO_Statemon_Coupled_STDP.Vs
Output_Coupled.IO.spikes = IO_spikes
del IO_spikes
Output_Coupled.IO.firingratesmooth = IO_rate_Coupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del IO_rate_Coupled_STDP
#####################################################################
################### DEEP CEREBELLAR NUCLEI CELLS ####################
#####################################################################
DCN_spikes = []
for i in range(0,DCN_Spikemon_Coupled_STDP.values('t').__len__(),1):
    DCN_spikes.append(DCN_Spikemon_Coupled_STDP.values('t')[i])
Output_Coupled.DCN.v = DCN_Statemon_Coupled_STDP.v
#         del DCN_Statemon_Coupled_STDP.v
Output_Coupled.DCN.spikes = DCN_spikes
del DCN_spikes
Output_Coupled.DCN.firingratesmooth = DCN_rate_Coupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del DCN_rate_Coupled_STDP
#####################################################################
############################## SAVING ###############################
#####################################################################
import numpy, scipy.io
local = datetime.datetime.now()
scipy.io.savemat('MyData_Coupled_Plasticity_'+str(noise_seed)+'.mat', mdict={'Output_Coupled': Output_Coupled})  
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
for i in range(0,PC_Spikemon_Uncoupled_STDP.values('t').__len__(),1):
    PC_spikes.append(PC_Spikemon_Uncoupled_STDP.values('t')[i])
Output_Uncoupled.PC.V = PC_Statemon_Uncoupled_STDP.v
#         del PC_Statemon_Uncoupled_STDP.v
Output_Uncoupled.PC.spikes = PC_spikes
del PC_spikes
Output_Uncoupled.PC.noise = PC_Statemon_Uncoupled_STDP.I_Noise
#         del PC_Statemon_Uncoupled_STDP.I_Noise
Output_Uncoupled.PC.firingratesmooth = PC_rate_Uncoupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del PC_rate_Uncoupled_STDP
Output_Uncoupled.PC.noise_source = mon_N_PC_Uncoupled.noise_source
#         del mon_N_PC_Uncoupled.noise_source
Output_Uncoupled.PC.PC_target = mon_N_PC_Uncoupled.PC_target
#         del mon_N_PC_Uncoupled.PC_target
Output_Uncoupled.PC.weight = mon_N_PC_Uncoupled.weight
#         del mon_N_PC_Uncoupled.weight
Output_Uncoupled.PC.I = mon_N_PC_Uncoupled.I
#         del mon_N_PC_Uncoupled.I
Output_Uncoupled.PC.new_weight = mon_N_PC_Uncoupled.new_weight
#         del mon_N_PC_Uncoupled.new_weight
Output_Uncoupled.PC.delta_weight = mon_N_PC_Uncoupled.delta_weight
#         del mon_N_PC_Uncoupled.delta_weight
Output_Uncoupled.PC.a_PC = mon_N_PC_Uncoupled.a_PC
#         del mon_N_PC_Uncoupled.a_PC
Output_Uncoupled.PC.a_IO = mon_N_PC_Uncoupled.a_IO
del mon_N_PC_Uncoupled
#####################################################################
###################### INFERIOR OLIVARY CELLS #######################
#####################################################################
IO_spikes = []
for i in range(0,IO_Spikemon_Uncoupled_STDP.values('t').__len__(),1):
    IO_spikes.append(IO_Spikemon_Uncoupled_STDP.values('t')[i])
Output_Uncoupled.IO.Vs = IO_Statemon_Uncoupled_STDP.Vs
#         del IO_Statemon_Uncoupled_STDP.Vs
Output_Uncoupled.IO.spikes = IO_spikes
del IO_spikes
Output_Uncoupled.IO.firingratesmooth = IO_rate_Uncoupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del IO_rate_Uncoupled_STDP
#####################################################################
################### DEEP CEREBELLAR NUCLEI CELLS ####################
#####################################################################
DCN_spikes = []
for i in range(0,DCN_Spikemon_Uncoupled_STDP.values('t').__len__(),1):
    DCN_spikes.append(DCN_Spikemon_Uncoupled_STDP.values('t')[i])
Output_Uncoupled.DCN.v = DCN_Statemon_Uncoupled_STDP.v
#         del DCN_Statemon_Uncoupled_STDP.v
Output_Uncoupled.DCN.spikes = DCN_spikes
del DCN_spikes
Output_Uncoupled.DCN.firingratesmooth = DCN_rate_Uncoupled_STDP.smooth_rate(window='gaussian', width=30*ms)/Hz
del DCN_rate_Uncoupled_STDP
#####################################################################
############################## SAVING ###############################
#####################################################################
import numpy, scipy.io
local = datetime.datetime.now()
scipy.io.savemat('MyData_Uncoupled_Plasticity_'+str(noise_seed)+'.mat', mdict={'Output_Uncoupled': Output_Uncoupled}) 