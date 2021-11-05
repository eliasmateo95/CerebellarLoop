from General_functions import *

def Noise_run(N_Noise,tau_noise,exp_run,dt,dt_rec):
    eqs_noise = '''
    dI/dt = (I0 - I)/tau_noise + sigma*xi*tau_noise**-0.5 : amp 
    I0 : amp
    sigma : amp
    weight : 1
    '''

    Noise = NeuronGroup(N_Noise, eqs_noise, threshold = 'True', method='euler',dt=dt)
    Noise_statemon = StateMonitor(Noise, variables=['I','weight'], record=True,dt=dt)

    Noise.I0 = 1.5*nA #rand_params(1.5,nA,N_noise,0.4)
    Noise.I = 1.5*nA #rand_params(1.5,nA,N_noise,0.3)
    Noise.sigma = 0.5*nA #rand_params(0.5,nA,N_noise,-0.3)

    run(exp_run,report='text')

    class Struct:
        pass
    Noise_Struct = Struct()

    Noise_Struct.time = Noise_statemon.t/ms
    Noise_Struct.I = Noise_statemon.I

    Noise_t = Noise_statemon.t/ms
    Noise_I = Noise_statemon.I
    Noise_I = numpy.ascontiguousarray(Noise_I, dtype=np.float64)
    N_Noise = len(Noise_I)
    
    return Noise_I


def Noise_neuron(N_Noise,I_recorded,exp_run,dt,dt_rec):
    eqs_noise = '''
    I = I_recorded(t,i)*amp : amp
    weight : 1 
    '''
    Noise = NeuronGroup(N_Noise, eqs_noise, threshold = 'True', method='euler',dt=dt)
    Noise_statemon = StateMonitor(Noise, variables=['I','weight'], record=True, dt=dt_rec)
    
    period = exp_run
    eqs_noise_extended = '''
    I = I_recorded(t % period,i)*amp : amp
    weight : 1 
    '''
    Noise_extended = NeuronGroup(N_Noise, eqs_noise_extended, threshold = 'True', method='euler',dt=dt)
    Noise_extended_statemon = StateMonitor(Noise_extended, variables=['I','weight'], record=True, dt=dt_rec)
    
    return Noise, Noise_statemon, Noise_extended, Noise_extended_statemon

def Noise_plot_neurons(N_Noise, Noise_statemon, dt_rec):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Noise):
        plot(Noise_statemon.t/ms,Noise_statemon.I[ii]/nA, label='Noise_Input'+str(ii+1))
    title('Current Input for '+str(N_Noise)+" Noise Sources")
    ylabel('I [nA]')
    xlabel('Time [ms]')
    legend()
    show()