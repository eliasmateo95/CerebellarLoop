from General_functions import *

def PC_DCN_syn(PC,DCN,N_Cells_PC,N_Cells_DCN,dt,dt_rec):
    PC_DCN_Synapse = Synapses(PC, DCN, on_pre='I_PC_post = 1.0*nA', delay=2*ms,dt=dt) 
    return PC_DCN_Synapse
    
def PC_DCN_Sources(N_Cells_PC,N_Cells_DCN):
    PC_DCN_Synapse_Sources = []
    PC_DCN_Synapse_Targets = []
    for pp in range(0,N_Cells_PC):
        PC_DCN_Synapse_Sources += N_Cells_PC * [pp]
        i = 0
        bb=[]
        while i < N_Cells_PC:
            r = randint(0,N_Cells_DCN)
            if r not in bb: 
                bb.append(r)
                i +=1
            else: 
                i = i
        PC_DCN_Synapse_Targets += bb 
    return PC_DCN_Synapse_Sources, PC_DCN_Synapse_Targets


