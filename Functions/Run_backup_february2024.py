from Input import *
from Output import *


def path_names_run(plasticity,path_data,exp_run,N_Cells_PC,N_Cells_DCN,N_Cells_IO,seed_number):
    net_path = path_data+'Frozen/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    seed_path = net_path+'/Seed_'+str(seed_number)
    run_path = seed_path+'/'+str(int(exp_run/msecond))+'ms'
    frozen_path = run_path+'/Frozen_'+str(int(exp_run/msecond))+'ms_'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'+'_Seed_'+str(seed_number)+'.mat'
    save_path_net = path_data+'Simulations/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    save_path_seed = save_path_net+'/Seed_'+str(seed_number)
    save_path_run = save_path_seed+'/'+str(int(exp_run/msecond))+'ms'
    save_path_plasticity = save_path_run+"/"+plasticity
    if os.path.exists(save_path_net) == False: os.mkdir(save_path_net)
    if os.path.exists(save_path_seed) == False: os.mkdir(save_path_seed)
    if os.path.exists(save_path_run) == False: os.mkdir(save_path_run)    
    if os.path.exists(save_path_plasticity) == False: os.mkdir(save_path_plasticity)    

    return net_path,seed_path,run_path,frozen_path,save_path_net,save_path_seed,save_path_run,save_path_plasticity



def run_me(parameters_value,plasticity_range,exp_run_range,network_range,zebrin_range,noise_gain_range,record,seed_number_range,path_data,save,show):
    f0_range = [0]
    if parameters_value["filtered"]:
        if parameters_value["unfiltered"]:
            f0_range.extend(parameters_value["f0_range"])
        else:
            f0_range = parameters_value["f0_range"]
    for f0, seed_number, noise_gain, zebrin, plasticity, N_Cells_PC, exp_run  in itertools.product(f0_range,seed_number_range,noise_gain_range,zebrin_range, plasticity_range,network_range, exp_run_range):

        time_start = time.monotonic()
        def time_checkpoint(name):
            nonlocal time_start
            e = time.monotonic()
            t = e - time_start
            time_start = e
            print(f'TIME: {name} took {t:.3f}s')
        time_checkpoint('starting')
        if N_Cells_PC == 100:
            N_Cells_DCN = 40
            N_Cells_IO = 40
        else:
            N_Cells_DCN = 2*N_Cells_PC
            N_Cells_IO = 2*N_Cells_PC   
        net_path,seed_path,run_path,frozen_path,save_path_net,save_path_seed,save_path_run,save_path_plasticity = path_names_run(plasticity,path_data,exp_run,N_Cells_PC,N_Cells_DCN,N_Cells_IO,seed_number)
        
        b2.device.reinit()
        b2.device.activate()
#         b2.set_device('cpp_standalone', debug=True)
        b2.set_device('cpp_standalone', directory=f'/mnt/Data/Elias/CerebellarLoop/Code/CPP/CPP_Standalone_output/{seed_number}', debug=True)
#         f0 = 0
        run_me_function(parameters_value,plasticity,zebrin,noise_gain,record,frozen_path,path_data,seed_number,f0,save,show)
        time_checkpoint(f'Seed: {seed_number}, Noise Gain: {noise_gain}, Zebrin: {zebrin}, Plasticity: {plasticity}, f0: {f0}')
#         response = webhook.send(text=f"Finished simulation seed: {seed_number}, {plasticity}, zebrin {zebrin}, f0: {f0}")
#         assert response.status_code == 200
#         assert response.body == "ok"
        b2.device.delete(force=True)
#     response = webhook.send(text="Finished simulation")
#     assert response.status_code == 200
#     assert response.body == "ok"

def run_me_function(parameters_value,plasticity,zebrin,noise_gain,record,frozen_path,path_data,seed_number,f0,save,show):
    time_start = time.monotonic()
    def time_checkpoint(name):
        nonlocal time_start
        e = time.monotonic()
        t = e - time_start
        time_start = e
        print(f'TIME: {name} took {t:.3f}s')
    start_scope()
    ###################################################################
    ######################### Load Parameters #########################
    ###################################################################
    Frozen_data = sio.loadmat(frozen_path, squeeze_me=True)
    Params, Noise_frozen, Values, Synaps = Read_Input(Frozen_data)
    Noise_record = record['Noise']
    PC_record = record['PC'] 
    DCN_record = record['DCN']
    IO_record = record['IO'] 
    conn_N_PC_record = record['conn_N_PC'] 
    Input_presyn_record = record['Input_presyn'] 
    ###################################################################
    ######################## Initial Parameters #######################
    ###################################################################
    dt = Params.dt
    dt_rec = Params.dt_rec
    width = Params.width
    tau_noise = Params.tau_noise
    exp_run = Params.exp_run
    N_Noise = Params.N_Noise
    N_Cells_PC = Params.N_Cells_PC
    N_Cells_DCN = Params.N_Cells_DCN
    N_Cells_IO = Params.N_Cells_IO
    N_Copy = Params.N_Copy
    N_Copy_order = Params.N_Copy_order
    ###################################################################
    ########################## Cell Values ############################
    ###################################################################
    Noise_I = Noise_frozen.Noise_I
    I_recorded = Noise_frozen.I_recorded
    if parameters_value['filtered']:
        if parameters_value["unfiltered"]:
            if f0 == 0:
                I_recorded = Noise_frozen.I_recorded    
            else:
                I_recorded = TimedArray(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}'].T, dt=dt_rec) #TimedArray(np.array(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}']).T, dt=dt_rec)
        else:
            I_recorded = TimedArray(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}'].T, dt=dt_rec) #TimedArray(np.array(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}']).T, dt=dt_rec)
    period = Noise_frozen.period
    PC_Values = Values.PC_Values
    unfiltered = parameters_value['unfiltered']
    filtered = parameters_value['filtered']
    name_check = f'unfiltered: {unfiltered}, filtered: {filtered}, f0:{f0}, Noise loaded'
    time_checkpoint(name_check)
#     if zebrin == 'positive':
#         zebrin_ff = 1.5-parameters_value["PC_I_intrinsic"]
#     elif zebrin == 'negative':
#         zebrin_ff = 1.5+parameters_value["PC_I_intrinsic"]
#     else:
#         zebrin_ff = 1.0
#     PC_Values["I_intrinsic"] = rand_params(zebrin_ff ,1,N_Cells_PC,(Values.PC_variablity/N_Cells_PC))  #[2*nA]*N_Cells_PC 
# #     print(PC_Values["PC_I_intrinsic"])
    PC_Values = Values.PC_Values
    PC_I_intrinsic = rand_params(parameters_value["PC_I_intrinsic"],1,N_Cells_PC,(Values.PC_variablity/N_Cells_PC))
    PC_I_intrinsic.sort()
    if zebrin == 'positive':
        zebrin_ff = np.array([0.2]*len(PC_I_intrinsic))-PC_I_intrinsic[::-1] #1.5-parameters_value["PC_I_intrinsic"]
    elif zebrin == 'negative':
        zebrin_ff = np.array([0.2]*len(PC_I_intrinsic))+PC_I_intrinsic #1.5+parameters_value["PC_I_intrinsic"]
    else:
        zebrin_ff = [0.0]*N_Cells_PC
#     print(zebrin_ff)
    PC_Values["I_intrinsic"] = list(zebrin_ff) # rand_params(zebrin_ff ,1,N_Cells_PC,(Values.PC_variablity/N_Cells_PC))  #[2*nA]*N_Cells_PC 
    DCN_Values = Values.DCN_Values
    DCN_Values["I_intrinsic"] = rand_params(1.75 ,1,N_Cells_DCN,(Values.DCN_variablity/N_Cells_DCN))  #[2*nA]*N_Cells_PC 
    IO_Values = Values.IO_Values
    IO_thresh = Values.IO_thresh
    eqs_IO_syn = Values.eqs_IO_syn
    rate_meas = Values.rate_meas
    rate_meas_out = Values.rate_meas_out
    rate_meas_PC = Values.rate_meas_PC
    rate_meas_out_PC = Values.rate_meas_out_PC
    tau_presyn = Values.tau_presyn

#     tau_thresh_M_val = 15*ms #int(seed_number.replace('New_BCM_debug_tanh_tau_', ''))*ms
#     tau_thresh_M_val = int(seed_number.replace('New_BCM_tanh_new_weight_tau_', ''))*ms
#     tau_thresh_M_val = int(seed_number.replace('Soti_IO_New_BCM_tanh_new_weight_tau_', ''))*ms
#     tau_thresh_M_val = int(seed_number.replace('New_BCM_tanh_new_weight_new_theta_tau_', ''))*ms
    tau_thresh_M_val = int(15)*ms
    
    
    eqs_syn_bcm_s_n_pc = Values.eqs_syn_bcm_s_n_pc
    eqs_syn_IO_PC_pre = Values.eqs_syn_IO_PC_pre
    w_IO_DCN = parameters_value[zebrin]['Uncoupled']['w_IO_DCN']
    w_IO_DCN_Coupled = parameters_value[zebrin]['Coupled']['w_IO_DCN']
    gCal_in = parameters_value[zebrin]['Uncoupled']['gCal']
    gCal_in_Coupled = parameters_value[zebrin]['Coupled']['gCal']
    ###################################################################
    ###################### Synapses Values ############################
    ###################################################################
    Synapses_conn = Synaps.Synapses
    IO_Copy_Synapse_Sources = Synaps.IO_Copy_Synapse_Sources
    Noise_PC_Synapse_Sources = Synaps.Noise_PC_Synapse_Sources
    Noise_PC_Synapse_Targets = Synaps.Noise_PC_Synapse_Targets 
    Noise_gain = noise_gain    
    if plasticity in parameters_value["range_after_plasticity"]:
#         print(plasticity)
        plasticity_before = plasticity.replace('after_', '')
        net_name = str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
        for coupling in ['','_Coupled']:
            savgol_filter_mon_N_PC_new_weight = output_load_run('mon_N_PC',coupling,seed_number,plasticity_before,zebrin,noise_gain,exp_run,net_name,path_data,parameters_value,f0)['new_weight']
            for col_syn_fil in range(0,len(savgol_filter_mon_N_PC_new_weight)):
                t = linspace(0,int(exp_run/second),len(savgol_filter_mon_N_PC_new_weight[col_syn_fil])) 
                signala = savgol_filter_mon_N_PC_new_weight[col_syn_fil] # with frequency of 100
                output_savgol = signal.savgol_filter(signala, 100, 2)
                savgol_filter_mon_N_PC_new_weight[col_syn_fil] = output_savgol
            if coupling == '_Coupled':
                Noise_PC_Synapse_Weights_Coupled = savgol_filter_mon_N_PC_new_weight[:,-1]
            else:
                Noise_PC_Synapse_Weights = savgol_filter_mon_N_PC_new_weight[:,-1]  
    else:
        Noise_PC_Synapse_Weights = Synaps.Noise_PC_Synapse_Weights
        Noise_PC_Synapse_Weights_Coupled = Synaps.Noise_PC_Synapse_Weights
    PC_DCN_Synapse_Sources = Synaps.PC_DCN_Synapse_Sources
    PC_DCN_Synapse_Targets = Synaps.PC_DCN_Synapse_Targets
    DCN_IO_Synapse_Sources = Synaps.DCN_IO_Synapse_Sources 
    DCN_IO_Synapse_Targets = Synaps.DCN_IO_Synapse_Targets 
    IO_PC_Synapse_Sources = Synaps.IO_PC_Synapse_Sources
    IO_PC_Synapse_Targets = Synaps.IO_PC_Synapse_Targets    
    if zebrin == 'positive':
        eqs_pc_dcn = 'I_PC_post += 0.005*nA'
    elif zebrin == 'negative':
        eqs_pc_dcn = 'I_PC_post += 0.005*nA'
    ###################################################################
    ###################################################################
    ############################## CELLS ##############################
    ###################################################################
    ###################################################################
    ############################## NOISE ##############################
    ###################################################################
    Noise, Noise_statemon = Noise_neuron(Noise_record,N_Noise,I_recorded,noise_gain,period,dt,dt_rec)
    ###################################################################
    ########################## PURKINJE CELL ##########################
    ###################################################################
    PC, PC_Statemon, PC_Spikemon, PC_rate = PC_neurons(PC_record,N_Cells_PC,PC_Values,dt,dt_rec)
    ###################################################################
    ################## DEEP CEREBELLAR NUCLEI CELLS ###################
    ###################################################################
    DCN, DCN_Statemon, DCN_Spikemon, DCN_rate = DCN_neurons(DCN_record,N_Cells_DCN,DCN_Values,dt,dt_rec)
    ###################################################################
    ############################# IO ##################################
    ###################################################################
    IO, IO_Statemon, IO_Spikemon, IO_rate = IO_neurons(IO_record,N_Cells_IO,IO_Values,IO_thresh,dt,dt_rec)
    ###################################################################
    ############################# PF ##################################
    ###################################################################
    Input_presyn, Input_presyn_statemon = presyn_inp(Input_presyn_record,I_recorded,plasticity,parameters_value["range_plasticity"],N_Noise, dt, dt_rec)
    ###################################################################
    ############################# Rates ###############################
    ###################################################################
    syn = rate_meas_func(rate_meas,PC,dt)
    syn.connect(j='i')  
    syn.subtract.delay = rate_meas_out  # delay the subtraction
    syn_PC = rate_meas_PC_func(rate_meas_PC,PC,dt)
    syn_PC.connect(j='i')  
    syn_PC.subtract.delay = rate_meas_out_PC  # delay the subtraction
    ###################################################################
    ############################# Copy ################################
    ###################################################################
    conn_N_PC, mon_N_PC = conn_N_PC_func(conn_N_PC_record,plasticity,parameters_value["range_plasticity"],N_Copy,Noise_PC_Synapse_Weights,dt,dt_rec)
    ###################################################################
    ###################################################################
    ############################## CELLS COUPLED ######################
    ###################################################################
    ###################################################################
    ############################## NOISE COUPLED ######################
    ###################################################################
    Noise_Coupled, Noise_statemon_Coupled = Noise_neuron(Noise_record,N_Noise,I_recorded,noise_gain,period,dt,dt_rec)
    ###################################################################
    ########################## PURKINJE CELL COUPLED ##################
    ###################################################################
    PC_Coupled, PC_Statemon_Coupled, PC_Spikemon_Coupled, PC_rate_Coupled = PC_neurons(PC_record,N_Cells_PC,PC_Values,dt,dt_rec)
    ###################################################################
    ################ DEEP CEREBELLAR NUCLEI CELLS COUPLED #############
    ###################################################################
    DCN_Coupled, DCN_Statemon_Coupled, DCN_Spikemon_Coupled, DCN_rate_Coupled = DCN_neurons(DCN_record,N_Cells_DCN,DCN_Values,dt,dt_rec)
    ###################################################################
    ############################# IO COUPLED ##########################
    ###################################################################
    IO_Coupled, IO_Statemon_Coupled, IO_Spikemon_Coupled, IO_rate_Coupled = IO_neurons(IO_record,N_Cells_IO,IO_Values,IO_thresh,dt,dt_rec)
    IO_synapse_Coupled = IO_coup_syn(IO_Coupled,eqs_IO_syn) # create synaptic equations and apply full synaptic strength for second network
    IO_synapse_Coupled.connect() # connect second network
    ###################################################################
    ############################# PF COUPLED ##########################
    ###################################################################
    Input_presyn_Coupled, Input_presyn_statemon_Coupled = presyn_inp(Input_presyn_record,I_recorded,plasticity,parameters_value["range_plasticity"],N_Noise, dt, dt_rec)
    ###################################################################
    ############################# Rates COUPLED #######################
    ###################################################################
    syn_Coupled = rate_meas_func(rate_meas,PC_Coupled,dt)
    syn_Coupled.connect(j='i')  
    syn_Coupled.subtract.delay = rate_meas_out  # delay the subtraction
    syn_PC_Coupled = rate_meas_PC_func(rate_meas_PC,PC_Coupled,dt)
    syn_PC_Coupled.connect(j='i')  
    syn_PC_Coupled.subtract.delay = rate_meas_out_PC  # delay the subtraction
    ###################################################################
    ############################# Copy COUPLED ########################
    ###################################################################
    conn_N_PC_Coupled, mon_N_PC_Coupled = conn_N_PC_func(conn_N_PC_record,plasticity,parameters_value["range_plasticity"],N_Copy, Noise_PC_Synapse_Weights_Coupled, dt, dt_rec)
    ###################################################################
    ###################################################################
    ########################## SYNAPSES ###############################
    ###################################################################
#     if Synapses_conn == True:
#         print(Synapses_conn)
    ###################################################################
    ########################## PC DCN Synapse #########################
    ###################################################################
    PC_DCN_Synapse = PC_DCN_syn(PC,DCN,N_Cells_PC,N_Cells_DCN,eqs_pc_dcn,dt,dt_rec)
    PC_DCN_Synapse.connect(i=PC_DCN_Synapse_Sources,j=PC_DCN_Synapse_Targets)
    ###################################################################
    ########################## DCN IO Synapse #########################
    ###################################################################
    w_IO_DCN_val = rand_params(w_IO_DCN,1,N_Cells_DCN,(0.1/N_Cells_DCN))
#         print(w_IO_DCN_val)
    DCN_IO_Synapse = DCN_IO_syn(DCN,IO,N_Cells_DCN,N_Cells_IO,w_IO_DCN_val,dt,dt_rec)
    DCN_IO_Synapse.connect(i=DCN_IO_Synapse_Sources,j=DCN_IO_Synapse_Targets)
    ###################################################################
    ########################### IO PC Synapse #########################
    ###################################################################
    IO_PC_Synapse = Synapses(IO, PC, on_pre = eqs_syn_IO_PC_pre, delay=2*ms,method = 'euler',dt=dt)
    IO_PC_Synapse.connect(i=IO_PC_Synapse_Sources,j=IO_PC_Synapse_Targets)
    ###################################################################
    ######################### ConnPC PC Synapse #######################
    ###################################################################
    S_N_PC = Synapses(conn_N_PC,PC, eqs_syn_bcm_s_n_pc, method='euler',dt=dt)
    S_N_PC.connect(i=N_Copy_order, j = Noise_PC_Synapse_Targets)
    ###################################################################
    ########################## ConnPC Noise Synapse ###################
    ###################################################################
    S_PC_N = Synapses(conn_N_PC,Noise, 'weight_post = new_weight_pre : 1 (summed)', method='euler',dt=dt)
    S_PC_N.connect(i=N_Copy_order, j = Noise_PC_Synapse_Sources)
    ###################################################################
    ############################# Copy rate ###########################
    ###################################################################
    copy_rate = Synapses(Input_presyn, conn_N_PC, 'rho_PF_post = abs(rho_presyn_pre-1.3*Hz) : Hz (summed)', method='euler',dt=dt)
    copy_rate.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
    ###################################################################
    ############################ Copy Noise ###########################
    ###################################################################
    copy_noise = Synapses(Noise, conn_N_PC, 'I_post = I_pre : amp (summed)', method='euler', dt=dt)
    copy_noise.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
    copy_noise.weight = Noise_PC_Synapse_Weights
    ###################################################################
    ########################## PC Rate Synapse ########################
    ###################################################################
    S_PC_rate = Synapses(PC,conn_N_PC, 'rho_PC_post = New_recent_rate_pre : Hz (summed)', method='euler',dt=dt)
    S_PC_rate.connect(i=Noise_PC_Synapse_Targets, j =N_Copy_order)
    ###################################################################
    ########################## IO ConnPC Synapse ######################
    ###################################################################
    if plasticity in parameters_value["range_plasticity"]:
        S_IO_N = Synapses(IO, conn_N_PC, on_pre = f'delta_weight_CS += {parameters_value["delta_weight_CS"]}*abs(rho_PF_post/Hz-1.3)', method='euler',dt=dt)  # where f is 
    else:
        S_IO_N = Synapses(IO, conn_N_PC, method='euler',dt=dt)  # where f is 
    S_IO_N.connect(i=IO_Copy_Synapse_Sources, j=N_Copy_order)
    ###################################################################
    ###################################################################
    ####################### SYNAPSES COUPLED ##########################
    ###################################################################
    ###################################################################
    ########################## PC DCN Synapse #########################
    ###################################################################
    PC_DCN_Synapse_Coupled = PC_DCN_syn(PC_Coupled,DCN_Coupled,N_Cells_PC,N_Cells_DCN,eqs_pc_dcn,dt,dt_rec)
    PC_DCN_Synapse_Coupled.connect(i=PC_DCN_Synapse_Sources,j=PC_DCN_Synapse_Targets)
    ###################################################################
    #################### DCN IO Synapse COUPLED #######################
    ###################################################################
    w_IO_DCN_val_Coupled = rand_params(w_IO_DCN_Coupled,1,N_Cells_DCN,(0.1/N_Cells_DCN))
    DCN_IO_Synapse_Coupled = DCN_IO_syn(DCN_Coupled,IO_Coupled,N_Cells_DCN,N_Cells_IO,w_IO_DCN_val_Coupled,dt,dt_rec)
    DCN_IO_Synapse_Coupled.connect(i=DCN_IO_Synapse_Sources,j=DCN_IO_Synapse_Targets)
    ###################################################################
    ##################### IO PC Synapse COUPLED #######################
    ###################################################################
    IO_PC_Synapse_Coupled = Synapses(IO_Coupled, PC_Coupled, on_pre =eqs_syn_IO_PC_pre, delay=2*ms,method = 'euler',dt=dt)
    IO_PC_Synapse_Coupled.connect(i=IO_PC_Synapse_Sources,j=IO_PC_Synapse_Targets)
    ###################################################################
    ################# ConnPC PC Synapse COUPLED #######################
    ###################################################################
    S_N_PC_Coupled = Synapses(conn_N_PC_Coupled,PC_Coupled, eqs_syn_bcm_s_n_pc, method='euler',dt=dt)
    S_N_PC_Coupled.connect(i=N_Copy_order, j = Noise_PC_Synapse_Targets)
    ###################################################################
    ############## ConnPC Noise Synapse COUPLED #######################
    ###################################################################
    S_PC_N_Coupled = Synapses(conn_N_PC_Coupled,Noise_Coupled, 'weight_post = new_weight_pre : 1 (summed)', method='euler',dt=dt)
    S_PC_N_Coupled.connect(i=N_Copy_order, j = Noise_PC_Synapse_Sources)
    ###################################################################
    ######################### Copy rate COUPLED #######################
    ###################################################################
    copy_rate_Coupled = Synapses(Input_presyn_Coupled, conn_N_PC_Coupled, 'rho_PF_post = abs(rho_presyn_pre-1.3*Hz) : Hz (summed)', method='euler',dt=dt)
    copy_rate_Coupled.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
    ###################################################################
    ######################## Copy Noise COUPLED #######################
    ###################################################################
    copy_noise_Coupled = Synapses(Noise_Coupled, conn_N_PC_Coupled, 'I_post = I_pre : amp (summed)', method='euler', dt=dt)
    copy_noise_Coupled.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
    copy_noise_Coupled.weight = Noise_PC_Synapse_Weights_Coupled
    ###################################################################
    ################### PC Rate Synapse COUPLED #######################
    ###################################################################
    S_PC_rate_Coupled = Synapses(PC_Coupled,conn_N_PC_Coupled, 'rho_PC_post = New_recent_rate_pre : Hz (summed)', method='euler',dt=dt)
    S_PC_rate_Coupled.connect(i=Noise_PC_Synapse_Targets, j =N_Copy_order)
    ###################################################################
    ################# IO ConnPC Synapse COUPLED #######################
    ###################################################################
    if plasticity in parameters_value["range_plasticity"]:
        S_IO_N_Coupled = Synapses(IO_Coupled, conn_N_PC_Coupled, on_pre = f'delta_weight_CS += {parameters_value["delta_weight_CS"]}*abs(rho_PF_post/Hz-1.3)', method='euler',dt=dt)  # where f is      
    else:
        S_IO_N_Coupled = Synapses(IO_Coupled, conn_N_PC_Coupled, method='euler',dt=dt)  # where f is 
    S_IO_N_Coupled.connect(i=IO_Copy_Synapse_Sources, j=N_Copy_order)
    
   
    ###################################################################
    if plasticity in parameters_value["range_plasticity"]:
        conn_N_PC.thresh_M = parameters_value["thresh_M"]#100*Hz
        conn_N_PC.tau_thresh_M = tau_thresh_M_val
        conn_N_PC_Coupled.thresh_M = parameters_value["thresh_M"]#100*Hz
        conn_N_PC_Coupled.tau_thresh_M = tau_thresh_M_val
        
    IO.g_ls += 0.001*mS*cm**-2
    IO.g_ld += 0.001*mS*cm**-2 
    IO.g_la += 0.001*mS*cm**-2
    
    gCal_values = rand_params(gCal_in,1,N_Cells_IO,(0.05/N_Cells_IO))*mS/cm**2
    gCal_values_Coupled = rand_params(gCal_in_Coupled,1,N_Cells_IO,(0.05/N_Cells_IO))*mS/cm**2
    
    dcn_intr = 1.2 #rand_params(1.2,1,N_Cells_DCN,(0.05/N_Cells_DCN))
    
    DCN.I_intrinsic = [dcn_intr]*N_Cells_DCN*nA
    DCN_Coupled.I_intrinsic = [dcn_intr]*N_Cells_DCN*nA
    
    IO.g_Ca_l =  gCal_values
    IO_Coupled.g_Ca_l =  gCal_values_Coupled

    sigma_ou = [0.3]*N_Cells_IO
    sigma_ou_Coupled = [0.3]*N_Cells_IO
    b_OU = [-0.3]*N_Cells_IO
    
    IO.sigma_OU = sigma_ou*uA/cm**2
    IO_Coupled.sigma_OU = sigma_ou_Coupled*uA/cm**2

    IO.I_OU = b_OU*uA/cm**2
    IO.I0_OU = b_OU*uA/cm**2

    IO_Coupled.I_OU = b_OU*uA/cm**2
    IO_Coupled.I0_OU = b_OU*uA/cm**2

    ###################################################################
    ########################### RUN ###################################
    ###################################################################
    time_checkpoint('about to run')
    run(exp_run) # Report on the simulation
    time_checkpoint('finished run, starting to save')

    
    if save == True:
        save_path_net = path_data+'Simulations/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
        save_path_seed = save_path_net+'/Seed_'+str(seed_number)
        save_path_run = save_path_seed+'/'+str(int(exp_run/msecond))+'ms'
        save_path_plasticity = save_path_run+"/"+plasticity
        if os.path.exists(save_path_plasticity) == False: 
            os.mkdir(save_path_plasticity)
            print("Successfully created the directory %s " % save_path_plasticity) 
            
        save_path_zebrin = save_path_plasticity+'/Zebrin_'+zebrin
        if os.path.exists(save_path_zebrin) == False: 
            os.mkdir(save_path_zebrin)
            print("Successfully created the directory %s " % save_path_zebrin) 
            
        save_path_noise = save_path_zebrin +'/Noise_gain_'+str(int(noise_gain*10))       
        if os.path.exists(save_path_noise) == False: 
            os.mkdir(save_path_noise)
            print("Successfully created the directory %s " % save_path_noise) 
            
        last_path = save_path_noise
        time_checkpoint('created save folders until filtered-unfiltered')
            
        if parameters_value['filtered']:
            if parameters_value["unfiltered"]:
                if f0 == 0:
                    last_path = save_path_noise
                else:
                    save_path_noise_filtered = save_path_noise +f'/Noise_filtered_{f0}'       
                    if os.path.exists(save_path_noise_filtered) == False: 
                        os.mkdir(save_path_noise_filtered)
                        print("Successfully created the directory %s " % save_path_noise_filtered)
                    last_path = save_path_noise_filtered
                    time_checkpoint('unfiltered but f0 is not 0')
            else:
                save_path_noise_filtered = save_path_noise +f'/Noise_filtered_{f0}'       
                if os.path.exists(save_path_noise_filtered) == False: 
                    os.mkdir(save_path_noise_filtered)
                    print("Successfully created the directory %s " % save_path_noise_filtered)
                last_path = save_path_noise_filtered
                time_checkpoint('filtered but not unfiltered')
            
        save_path_fig = last_path+"/Figures"
        if os.path.exists(save_path_fig) == False: 
            os.mkdir(save_path_fig) 
            print("Successfully created the directory %s " % save_path_fig)
            
        save_path_data = last_path+"/Output"
        if os.path.exists(save_path_data) == False: 
            os.mkdir(save_path_data)
            print("Successfully created the directory %s " % save_path_data)
   

    Create_output(plasticity,zebrin,save_path_data,width,Noise_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Input_presyn_statemon,Noise_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled,Input_presyn_statemon_Coupled)