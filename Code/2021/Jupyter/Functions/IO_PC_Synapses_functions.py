from General_functions import *

def IO_PC_syn(IO,PC,N_Cells_IO,N_Cells_PC,dt,dt_rec):
    IO_PC_Synapse = Synapses(IO, PC, on_pre ='w +=(1*nA)', delay=2*ms,method = 'euler',dt=dt)
    return IO_PC_Synapse
    
def IO_PC_Sources(N_Cells_IO,N_Cells_PC):
    IO_PC_Synapse_Sources = []
    IO_PC_Synapse_Targets = []
    i = 0
    bb=[]
    while i < N_Cells_PC:
        r = randint(0,N_Cells_IO)
        if r not in bb:
            IO_PC_Synapse_Targets += [i]
            bb.append(r)
            i +=1
        else: 
            i = i
    IO_PC_Synapse_Sources += bb 
    return IO_PC_Synapse_Sources, IO_PC_Synapse_Targets

