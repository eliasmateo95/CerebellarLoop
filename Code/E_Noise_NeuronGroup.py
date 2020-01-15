from D_Cells_Values import *

#####################################################################
############################ Input Noise ############################
#####################################################################
N_Noise = len(Noise_I)
I_recorded = TimedArray(Noise_I.T, dt=dt)
eqs_noise = '''
I = I_recorded(t,i)*amp : amp
'''
Noise = NeuronGroup(N_Noise, eqs_noise, threshold = 'True', method='euler',name = 'Noise',dt=dt)
Noise_statemon = StateMonitor(Noise, variables=['I'], record=True, dt=dt_rec)

period = exp_runtime
eqs_noise_extended = '''
I = I_recorded(t % period,i)*amp : amp
'''
Noise_extended = NeuronGroup(N_Noise, eqs_noise_extended, threshold = 'True', method='euler',name = 'Noise_extended',dt=dt)
Noise_extended_statemon = StateMonitor(Noise_extended, variables=['I'], record=True, dt=dt_rec)
