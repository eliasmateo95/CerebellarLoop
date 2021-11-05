from General_functions import *

def New_PC(Params,PC):
    PC_new = [[]]*Params.N_Cells_PC
    for ii in range(0,Params.N_Cells_PC):
        vm = PC.v[ii]
        for t in PC.Spikemon_Cells[ii]:
            i = int(t / Params.dt_rec)
            vm[i] = 20*mV
        PC_new[ii] = vm
    return PC_new

def New_PC_learned(Params,PC):
    PC_new = [[]]*Params.N_Cells_PC
    for ii in range(0,Params.N_Cells_PC):
        vm = PC.v[ii]
        for t in PC.Spikemon_Cells[ii]:
            i = int(t / Params.dt_rec)
            vm[i] = 20*mV
        PC_new[ii] = vm
    PC_new_learned = [[]]*len(PC_new)
    for ii in range(0,len(PC_new)):
        PC_new_learned[ii] = PC_new[ii][len(PC_new[ii])//2:]
    return PC_new_learned


       
def CS_plots(step,learned,Params,Synapses,IO):
    kk = 0
    IO_CS = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        IO_CS[kk] = [[]]*size(IO_spikes)
        for jj in range(0,size(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            IO_CS[kk][jj] = (IO.Vs[ii][int(spike-step):int(spike+step)]/mV)
        kk += 1

    IO_Size = [[]]*len(IO_CS)
    for ii in range(0,len(IO_CS)):
        IO_Size[ii] = [[]]*len(IO_CS[ii])
        for jj in range(0,len(IO_CS[ii])):
            IO_Size[ii][jj] = len(IO_CS[ii][jj]) 

    kk = 0
    aa = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    bb = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    cc = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        aa[kk] = [[]]*size(IO_spikes)
        bb[kk] = [[]]*size(IO_spikes)
        for jj in range(0,size(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            aa[kk][jj] = IO.Vs[ii][int(spike-step):int(spike+step)]
            if size(aa[kk][jj]) != most_frequent(IO_Size[kk]):
                continue
            bb[kk][jj] = IO.Vs[ii][int(spike-step):int(spike+step)]
        cc[kk] = [x for x in bb[kk] if x != []]    
        kk+=1

    avrg_CS = [[]]*len(cc)
    for ii in range(0,len(cc)):
        avrg_CS[ii] = np.mean(cc[ii], axis=0)
   
    avrg_CS_all = np.mean(avrg_CS, axis=0)
        
    return IO_CS, IO_Size, cc, avrg_CS, avrg_CS_all

def CS_PC(step,learned,Params,Synapses,PC_new,IO):
    kk = 0

    PC_CS = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        PC_CS[kk] = [[]]*len(IO_spikes)
        for jj in range(0,len(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            if int(spike-step) < 0:
                continue
            PC_CS[kk][jj] = (PC_new[kk][int(spike-step):int(spike+step)]/mV)
        kk += 1

    PC_Size = [[]]*len(PC_CS)
    for ii in range(0,len(PC_CS)):
        PC_Size[ii] = [[]]*len(PC_CS[ii])
        for jj in range(0,len(PC_CS[ii])):
            PC_Size[ii][jj] = len(PC_CS[ii][jj]) 

    kk = 0
    aa = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    bb = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    cc = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        aa[kk] = [[]]*len(IO_spikes)
        bb[kk] = [[]]*len(IO_spikes)
        for jj in range(0,len(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            aa[kk][jj] = PC_new[kk][int(spike-step):int(spike+step)]
            if size(aa[kk][jj]) != most_frequent(PC_Size[kk]):
                continue
            bb[kk][jj] = PC_new[kk][int(spike-step):int(spike+step)]
        cc[kk] = [x for x in bb[kk] if x != []]    
        kk+=1

    avrg_PC = [[]]*len(cc)
    for ii in range(0,len(cc)):
        avrg_PC[ii] = np.mean(cc[ii], axis=0)

    avrg_PC_all = np.mean(avrg_PC, axis=0)
    
    return PC_CS, cc, avrg_PC, avrg_PC_all


def PC_CS_rate(step,learned,Params,Synapses,PC,IO):
    kk = 0
    PC_CS_rate = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        PC_CS_rate[kk] = [[]]*len(IO_spikes)
        for jj in range(0,len(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            if int(spike-step) < 0:
                continue
            PC_CS_rate[kk][jj] = (PC.rate[int(spike-step):int(spike+step)]/mV)
        kk += 1

    PC_Rate_Size = [[]]*len(PC_CS_rate)
    for ii in range(0,len(PC_CS_rate)):
        PC_Rate_Size[ii] = [[]]*len(PC_CS_rate[ii])
        for jj in range(0,len(PC_CS_rate[ii])):
            PC_Rate_Size[ii][jj] = len(PC_CS_rate[ii][jj]) 


    kk = 0
    aa = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    bb = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    cc_rate = [[]]*size(Synapses.IO_PC_Synapse_Sources)
    for ii in Synapses.IO_PC_Synapse_Sources:
        if learned == "Learned":
            IO_spikes = IO.Spikemon_Cells[ii][len(IO.Spikemon_Cells[ii])//2:]
        else: 
            IO_spikes = IO.Spikemon_Cells[ii][:len(IO.Spikemon_Cells[ii])//2]
        aa[kk] = [[]]*len(IO_spikes)
        bb[kk] = [[]]*len(IO_spikes)
        for jj in range(0,len(IO_spikes)):
            spike = IO_spikes[jj]/(Params.dt_rec/second)
            aa[kk][jj] = PC.rate[int(spike-step):int(spike+step)]
            if size(aa[kk][jj]) != most_frequent(PC_Rate_Size[kk]):
                continue
            bb[kk][jj] = PC.rate[int(spike-step):int(spike+step)]
        cc_rate[kk] = [x for x in bb[kk] if x != []]    
        kk+=1

    avrg_PC_rate = [[]]*len(cc_rate)
    for ii in range(0,len(cc_rate)):
        avrg_PC_rate[ii] = np.mean(cc_rate[ii], axis=0)

    avrg_PC_all_rate = np.mean(avrg_PC_rate, axis=0)

    return PC_CS_rate, cc_rate, avrg_PC_rate, avrg_PC_all_rate