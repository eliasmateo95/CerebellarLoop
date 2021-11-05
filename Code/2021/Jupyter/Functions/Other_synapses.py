from General_functions import *

def rate_meas_func(rate_meas,PC,dt):
    syn = Synapses(PC, PC, '', on_pre={'add': 'recent_rate += 1/rate_meas', 
                                             'subtract': 'recent_rate -= 1/rate_meas'}, method='euler', dt=dt)     
    return syn



def presyn_inp(N_Noise,sine_amplitude_presyn,sine_frequency_presyn, dt, dt_rec):
    eqs_presyn = '''
            rho_presyn = ampli+sine_amplitude_presyn*sin(sine_frequency_presyn*t): Hz
            sine_amplitude_presyn : Hz
            sine_frequency_presyn : Hz
            ampli : Hz
            '''
    Input_presyn = NeuronGroup(N_Noise, eqs_presyn, threshold='True', method='euler', dt=dt)
    Input_presyn_statemon = StateMonitor(Input_presyn, variables=['rho_presyn'], record=True, dt=dt_rec)
    
    ampli = rand_params(40,Hz,N_Noise,(5.0/N_Noise))
    tau_presyn = rand_params(10,ms,N_Noise,(1.0/N_Noise))
    sine_amplitude_presyn = rand_params(10,Hz,N_Noise,(1.0/N_Noise))
    for ii in range(0, N_Noise, 1):
        Input_presyn.ampli[ii] = ampli[ii]
        Input_presyn.sine_amplitude_presyn[ii] = sine_amplitude_presyn[ii]
        Input_presyn.sine_frequency_presyn[ii] = 1/tau_presyn[ii]

    return Input_presyn, Input_presyn_statemon

def conn_N_PC_func(N_Copy, Noise_PC_Synapse_Weights, dt, dt_rec):
    eqs_Copy = '''
                I : amp  # copy of the noise current
                rho_PF : Hz 
                rho_PC : Hz
                weight : 1 (constant)
                new_weight = weight + delta_weight : 1 
                ddelta_weight/dt = (rho_PC*(rho_PC-thresh_M)/thresh_M)*rho_PF*msecond : 1 
                phi = rho_PC*(rho_PC-thresh_M)/thresh_M : Hz 
                dthresh_M/dt = rho_PC**2 - thresh_M/tau_thresh_M : Hz 
    '''
    conn_N_PC = NeuronGroup(N_Copy, eqs_Copy, method='euler',dt=dt)
    conn_N_PC.weight = Noise_PC_Synapse_Weights
    mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','phi','thresh_M','delta_weight','new_weight'], record=True, dt=dt_rec)
    

    return conn_N_PC, mon_N_PC

def conn_N_PC_No_BCM_func(N_Copy, Noise_PC_Synapse_Weights, dt, dt_rec):
    eqs_Copy = '''
                I : amp  # copy of the noise current
                rho_PF : Hz 
                rho_PC : Hz
                weight : 1 (constant)
                new_weight = weight : 1 
    '''
    conn_N_PC = NeuronGroup(N_Copy, eqs_Copy, method='euler',dt=dt)
    conn_N_PC.weight = Noise_PC_Synapse_Weights
    mon_N_PC = StateMonitor(conn_N_PC , ['rho_PF','rho_PC','new_weight'], record=True, dt=dt_rec)


    return conn_N_PC, mon_N_PC