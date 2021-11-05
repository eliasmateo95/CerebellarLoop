from General_functions import *

def Elias_STDP(tau_PC,tau_IO,wmax,A_PC,A_IO,N_Noise,N_Cells_PC,N_Copy,PC,IO,Noise_extended,dt,dt_rec):
    eqs_syn_Noise_PC_STDP = '''
                            I : amp  # copy of the noise current
                            weight : 1 (constant)
                            new_weight = weight + delta_weight : 1 
                            delta_weight = a_PC + a_IO : 1  # Change of delta due to LTD/LTP
                            da_PC/dt = -a_PC/tau_PC : 1  # PC influence on weight
                            da_IO/dt = -a_IO/tau_IO : 1  # IO influence on weight
    '''
    eqs_syn_stdp_s_n_pc = '''
                          I_Noise_post = (new_weight_pre)*(I_pre)*(1.0/N_Noise) : amp (summed)
    '''

    conn_N_PC = NeuronGroup(N_Copy, eqs_syn_Noise_PC_STDP, method='euler',dt=dt)
    mon_N_PC = StateMonitor(conn_N_PC , ['a_PC','a_IO','I','delta_weight'], record=True, dt=dt_rec)
    
    
    copy_noise = Synapses(Noise_extended, conn_N_PC, 'I_post = I_pre : amp (summed)')

    # Synapses to Purkinje cells
    S_N_PC = Synapses(conn_N_PC, PC,eqs_syn_stdp_s_n_pc, on_post='a_PC_pre += A_PC', method='euler',dt=dt)

    # LTD from IO cells:
    S_IO_N = Synapses(IO, conn_N_PC, on_pre='a_IO_post += A_IO*(new_weight_post*I_post)/nA', method='euler',dt=dt)  # where f is some function

    return conn_N_PC, mon_N_PC, copy_noise, S_N_PC, S_IO_N
