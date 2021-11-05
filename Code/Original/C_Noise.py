from B_StartUp import *

tau_noise = 50*ms
eqs_noise = '''
dI/dt = (I0 - I)/tau_noise + sigma*xi*tau_noise**-0.5 : amp 
I0 : amp
sigma : amp
weight : 1
'''

Noise = NeuronGroup(N_noise, eqs_noise, threshold = 'True', method='euler',name = 'Noise',dt=dt)
Noise_statemon = StateMonitor(Noise, variables=['I','weight'], record=True,dt=dt)

Noise.I0 = 1.5*nA #rand_params(1.5,nA,N_noise,0.4)
Noise.I = 1.5*nA #rand_params(1.5,nA,N_noise,0.3)
Noise.sigma = 0.5*nA #rand_params(0.5,nA,N_noise,-0.3)

run(exp_runtime,report='text')

class Struct:
    pass
Noise = Struct()

Noise.time = Noise_statemon.t/ms
Noise.I = Noise_statemon.I

Noise_t = Noise_statemon.t/ms
Noise_I = Noise_statemon.I
Noise_I = numpy.ascontiguousarray(Noise_I, dtype=np.float64)
N_Noise = len(Noise_I)