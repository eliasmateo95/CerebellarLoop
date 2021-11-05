from H_Synapses_NoPlasticity import *
from H_Synapses_Plasticity import *

run(exp_runtime,report='text')
if saving == 'yes':   
    from J_Saving_NoPlasticity.py import *

run(exp_runtime,report='text')
if saving == 'yes':     
    from J_Saving_Plasticity import *

    
import numpy, scipy.io
noise_seed = noise_seed + 1
scipy.io.savemat('Noise_Seed.mat', mdict={'noise_seed': noise_seed})   
