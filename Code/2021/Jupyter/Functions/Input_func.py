from Cell_Values import *
from Noise_functions import *
from Noise_PC_Synapses_functions import *
from PC_DCN_Synapses_functions import *
from DCN_IO_Synapses_functions import *
from IO_PC_Synapses_functions import *

def Input_func(dt, dt_rec, tau_noise, exp_run, width, N_Noise, N_Cells_PC, N_Cells_DCN, N_Cells_IO, rate_meas, rate_meas_out, tau_presyn, tau_thresh_M, sine_amplitude_presyn, sine_frequency_presyn):
    Input = {}
    Input["Parameters"] = {}
    Input["Parameters"]["dt"] = dt #= 0.025*ms
    Input["Parameters"]["dt_rec"] = dt_rec #= 0.25*ms
    Input["Parameters"]["tau_noise"] = tau_noise #= 50*ms
    Input["Parameters"]["exp_run"] = exp_run #= 5000*ms
    Input["Parameters"]["width"] = width
    Input["Parameters"]["N_Noise"] = N_Noise #= 5
    Input["Parameters"]["N_Cells_PC"] = N_Cells_PC #= 5
    Input["Parameters"]["N_Cells_DCN"] = N_Cells_DCN #= 10
    Input["Parameters"]["N_Cells_IO"] = N_Cells_IO #= 10
    Input["Parameters"]["N_Copy"] = N_Copy = N_Noise*N_Cells_PC
    Input["Parameters"]["N_Copy_order"] = N_Copy_order = list(range(0,N_Copy))
    ########################## Cell Values ############################
    Input["Noise"] = {}
    Input["Noise"]["Noise_I"] = Noise_I = Noise_run(N_Noise,tau_noise,exp_run,dt,dt_rec)
    Input["PC"] = {}
    Input["PC"]["PC_Values"] = PC_cell_values(N_Cells_PC)
    Input["DCN"] = {}
    Input["DCN"]["DCN_Values"] = DCN_cell_values(N_Cells_DCN)
    Input["DCN"]["eqs_syn_IO_PC_pre"] = 'w +=(1.5*nA); try_new_bcm += 100*Hz'
    Input["IO"] = {}
    Input["IO"]["IO_Values"] = IO_cell_values(N_Cells_IO)
    Input["IO"]["IO_thresh"] = IO_thresh = 'Vs>20*mV'
    Input["IO"]["eqs_IO_syn"] = eqs_IO_syn = ''' I_c_pre = (0.00125*mS/cm**2)*(0.6*e**(-((Vd_pre/mvolt-Vd_post/mvolt)/50)**2) + 0.4)*(Vd_pre-Vd_post) : metre**-2*amp (summed)'''
    Input["Copy"] = {}
    Input["Copy"]["rate_meas"] = rate_meas
    Input["Copy"]["rate_meas_out"] = rate_meas_out
    Input["Copy"]["tau_presyn"] = tau_presyn
    Input["Copy"]["tau_thresh_M"] = tau_thresh_M
    Input["Copy"]["sine_amplitude_presyn"] = sine_amplitude_presyn
    Input["Copy"]["sine_frequency_presyn"] = sine_frequency_presyn
    Input["Copy"]["eqs_syn_bcm_s_n_pc"] = '''
                          I_Noise_empty_post = weight_pre*I_pre : amp (summed)
                          I_Noise_post = (new_weight_pre)*(I_pre)*(1.0/N_Noise) : amp (summed)
    '''
    ########################## Synapses ###############################
    Noise_PC_Synapse_Sources, Noise_PC_Synapse_Targets, Noise_PC_Synapse_Weights = Noise_PC_Weights(N_Noise,N_Cells_PC)
    PC_DCN_Synapse_Sources, PC_DCN_Synapse_Targets = PC_DCN_Sources(N_Cells_PC,N_Cells_DCN)
    DCN_IO_Synapse_Sources, DCN_IO_Synapse_Targets = DCN_IO_Sources(N_Cells_PC,N_Cells_DCN,N_Cells_IO)
    IO_PC_Synapse_Sources, IO_PC_Synapse_Targets = IO_PC_Sources(N_Cells_IO,N_Cells_PC)
    Input["Synapses"] = {}
    Input["Synapses"]["IO_Copy_Synapse_Targets"] = IO_Copy_Synapse_Targets = [x*N_Noise for x in IO_PC_Synapse_Targets]
    Input["Synapses"]["Noise_PC_Synapse_Sources"] = Noise_PC_Synapse_Sources
    Input["Synapses"]["Noise_PC_Synapse_Targets"] = Noise_PC_Synapse_Targets
    Input["Synapses"]["Noise_PC_Synapse_Weights"] = Noise_PC_Synapse_Weights
    Input["Synapses"]["PC_DCN_Synapse_Sources"] = PC_DCN_Synapse_Sources
    Input["Synapses"]["PC_DCN_Synapse_Targets"] = PC_DCN_Synapse_Targets
    Input["Synapses"]["DCN_IO_Synapse_Sources"] = DCN_IO_Synapse_Sources
    Input["Synapses"]["DCN_IO_Synapse_Targets"] = DCN_IO_Synapse_Targets
    Input["Synapses"]["IO_PC_Synapse_Sources"] = IO_PC_Synapse_Sources
    Input["Synapses"]["IO_PC_Synapse_Targets"] = IO_PC_Synapse_Targets
    
    path = os.getcwd()
    save_path = path+"/Data/"+datetime.datetime.now().strftime("%m-%d")
    try:
        os.mkdir(save_path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)    
    Name = ""
    file_name = 'Frozen.mat'
    completeName = os.path.join(save_path, file_name)

    sio.savemat(completeName,Input) 
    

def Read_Input(Frozen_data):
    class Noise:
        pass
    class Params:
        pass
    class Noise_frozen:
        pass
    class Values:
        pass
    class Synapses:
        pass
    Params.dt = dt = Frozen_data["Parameters"]["dt"]*second
    Params.dt_rec = Frozen_data["Parameters"]["dt_rec"]*second
    Params.tau_noise = Frozen_data["Parameters"]["tau_noise"]*second
    Params.exp_run = Frozen_data["Parameters"]["exp_run"]*second
    Params.width = Frozen_data["Parameters"]["width"]*second
    Params.N_Noise = Frozen_data["Parameters"]["N_Noise"].item()
    Params.N_Cells_PC = Frozen_data["Parameters"]["N_Cells_PC"].item()
    Params.N_Cells_DCN = Frozen_data["Parameters"]["N_Cells_DCN"].item()
    Params.N_Cells_IO = Frozen_data["Parameters"]["N_Cells_IO"].item()
    Params.N_Copy = Frozen_data["Parameters"]["N_Copy"].item()
    Params.N_Copy_order = Frozen_data["Parameters"]["N_Copy_order"].item()
    ########################## Cell Values ############################
    Noise_frozen.Noise_I = Noise_I = Frozen_data["Noise"]["Noise_I"].item()
    Noise_frozen.I_recorded = TimedArray(Noise_I.T, dt=dt)
    Noise_frozen.period = Params.exp_run

    Values.PC_Values = Frozen_data["PC"]["PC_Values"].item()
    Values.DCN_Values = Frozen_data["DCN"]["DCN_Values"].item()
    Values.IO_Values = Frozen_data["IO"]["IO_Values"].item()
    Values.IO_thresh = Frozen_data["IO"]["IO_thresh"].item()
    Values.eqs_IO_syn = Frozen_data["IO"]["eqs_IO_syn"].item()
    Values.rate_meas = Frozen_data["Copy"]["rate_meas"]*second
    Values.rate_meas_out = Frozen_data["Copy"]["rate_meas_out"]*second
    Values.tau_presyn = Frozen_data["Copy"]["tau_presyn"]*second
    Values.tau_thresh_M = Frozen_data["Copy"]["tau_thresh_M"]*second
    Values.sine_amplitude_presyn = Frozen_data["Copy"]["sine_amplitude_presyn"]*Hz
    Values.sine_frequency_presyn = Frozen_data["Copy"]["sine_frequency_presyn"]*Hz
    Values.eqs_syn_bcm_s_n_pc = Frozen_data["Copy"]["eqs_syn_bcm_s_n_pc"].item()
    Values.eqs_syn_IO_PC_pre = Frozen_data["DCN"]["eqs_syn_IO_PC_pre"].item()
    ########################## Synapses ###############################
    Synapses.IO_Copy_Synapse_Targets = Frozen_data["Synapses"]["IO_Copy_Synapse_Targets"].item()
    Synapses.Noise_PC_Synapse_Sources = Frozen_data["Synapses"]["Noise_PC_Synapse_Sources"].item()
    Synapses.Noise_PC_Synapse_Targets = Frozen_data["Synapses"]["Noise_PC_Synapse_Targets"].item()  
    Synapses.Noise_PC_Synapse_Weights = Frozen_data["Synapses"]["Noise_PC_Synapse_Weights"].item() 
    Synapses.PC_DCN_Synapse_Sources = Frozen_data["Synapses"]["PC_DCN_Synapse_Sources"].item()
    Synapses.PC_DCN_Synapse_Targets = Frozen_data["Synapses"]["PC_DCN_Synapse_Targets"].item()
    Synapses.DCN_IO_Synapse_Sources = Frozen_data["Synapses"]["DCN_IO_Synapse_Sources"].item() 
    Synapses.DCN_IO_Synapse_Targets = Frozen_data["Synapses"]["DCN_IO_Synapse_Targets"].item() 
    Synapses.IO_PC_Synapse_Sources = Frozen_data["Synapses"]["IO_PC_Synapse_Sources"].item()
    Synapses.IO_PC_Synapse_Targets = Frozen_data["Synapses"]["IO_PC_Synapse_Targets"].item()
    
    return Params, Noise_frozen, Values, Synapses