from General_functions import *

def DCN_IO_syn(DCN,IO,N_Cells_DCN,N_Cells_IO,dt,dt_rec):
    IO_DCN_Synapse = Synapses(DCN,IO,on_pre = 'I_IO_DCN_post += -0.1/N_Cells_DCN*uA*cm**-2', delay=3*ms, method = 'euler',dt=dt)
    return IO_DCN_Synapse
#     (20/(N_Cells_IO*(N_Cells_DCN/2.0)))
def DCN_IO_Sources(N_Cells_PC,N_Cells_DCN,N_Cells_IO):
    IO_DCN_Synapse_Sources = []
    IO_DCN_Synapse_Targets = []
    for pp in range(0,N_Cells_DCN):
        IO_DCN_Synapse_Sources += N_Cells_PC * [pp]
        i = 0
        bb=[]
        while i < N_Cells_PC:
            r = randint(0,N_Cells_IO)
            if r not in bb: 
                bb.append(r)
                i +=1
            else: 
                i = i
        IO_DCN_Synapse_Targets += bb 
    return IO_DCN_Synapse_Sources, IO_DCN_Synapse_Targets

