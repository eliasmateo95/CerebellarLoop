from Input import *

def path_names_run(plasticity,path_data,exp_run,N_Cells_PC,N_Cells_DCN,N_Cells_IO,seed_number):
    net_path = path_data+'Frozen/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    seed_path = net_path+'/Seed_'+str(seed_number)
    run_path = seed_path+'/'+str(int(exp_run/msecond))+'ms'
    frozen_path = run_path+'/Frozen_'+str(int(exp_run/msecond))+'ms_'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'+'_Seed_'+str(seed_number)+'.mat'
    save_path_net = path_data+'/Simulations/Networks/'+str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
    save_path_seed = save_path_net+'/Seed_'+str(seed_number)
    save_path_run = save_path_seed+'/'+str(int(exp_run/msecond))+'ms'
    save_path_plasticity = save_path_run+"/"+plasticity
    if os.path.exists(save_path_net) == False: os.mkdir(save_path_net)
    if os.path.exists(save_path_seed) == False: os.mkdir(save_path_seed)
    if os.path.exists(save_path_run) == False: os.mkdir(save_path_run)    
    if os.path.exists(save_path_plasticity) == False: os.mkdir(save_path_plasticity)    

    return net_path,seed_path,run_path,frozen_path,save_path_net,save_path_seed,save_path_run,save_path_plasticity

  
    
def run_me(PS_params,plasticity_range,exp_run_range,network_range,zebrin_range,noise_gain_range,record,seed_number_range,path_data,save,show):
    for seed_number, noise_gain, gCal_val,w_IO_DCN_val,PC_DCN_val,leak_val,sigma_OU_val,b_OU_val,zebrin, plasticity, N_Cells_PC, exp_run  in itertools.product(seed_number_range,noise_gain_range,PS_params['gCal_range'],PS_params['w_IO_DCN_range'],PS_params['PC_DCN_range'],PS_params['leak_range'],PS_params['sigma_OU_range'],PS_params['b_OU_range'],zebrin_range, plasticity_range,network_range, exp_run_range):
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
        run_me_function(plasticity,zebrin,noise_gain,gCal_val,w_IO_DCN_val,PC_DCN_val,leak_val,sigma_OU_val,b_OU_val,record,frozen_path,path_data,seed_number,save_path_run,save,show)
        time_checkpoint(f'Seed: {seed_number}, Noise Gain: {noise_gain}, Zebrin: {zebrin}, Plasticity: {plasticity}')
        response = webhook.send(text=f"Finished simulation seed: {seed_number}, {plasticity}, zebrin {zebrin}")
        assert response.status_code == 200
        assert response.body == "ok"
        b2.device.delete(force=True)
    response = webhook.send(text="Finished simulation")
    assert response.status_code == 200
    assert response.body == "ok"
#         net_path,seed_path,run_path,frozen_path,save_path_net,save_path_seed,save_path_run,save_path_plasticity = path_names_run(plasticity,path_data,exp_run,N_Cells_PC,N_Cells_DCN,N_Cells_IO,seed_number)
       
#         b2.device.reinit()
#         b2.device.activate()
# #         b2.set_device('cpp_standalone', debug=True)
#         b2.set_device('cpp_standalone', directory=f'/mnt/Data/Elias/CerebellarLoop/Code/CPP/CPP_Standalone_output/{seed_number}', debug=True) 
#     run_me_function(plasticity,zebrin,noise_gain,gCal_val,w_IO_DCN_val,record,frozen_path,path_data,seed_number,save_path_run,save,show)
        
#         time_checkpoint(f'Seed: {seed_number}, Noise Gain: {noise_gain}, Zebrin: {zebrin}, Plasticity: {plasticity}')
#         response = webhook.send(text=f"Finished simulation seed: {seed_number}, {plasticity}, zebrin {zebrin}")
#         assert response.status_code == 200
#         assert response.body == "ok"
#         b2.device.delete(force=True)   
#     response = webhook.send(text="Finished simulation")
#     assert response.status_code == 200
#     assert response.body == "ok"

def run_me_function(plasticity,zebrin,noise_gain,gCal_val,w_IO_DCN_val,PC_DCN_val,leak_val,sigma_OU_val,b_OU_val,record,frozen_path,path_data,seed_number,save_path_run,save,show):
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
    f0 = 0
    parameters_value = { "range_plasticity": [], "range_after_plasticity": [],
                "PC_I_intrinsic": 0.15, "thresh_M": 0*Hz, "delta_weight_CS": -0.1,
                "unfiltered": True, "filtered": False, "f0_range": [], "filter_order": 6,
        'no_Plasticity': {'Uncoupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val},'Coupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val}},
        'Plasticity': {'Uncoupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val},'Coupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val}},
        'after_Plasticity': {'Uncoupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val},'Coupled': {'gCal': gCal_val, 'w_IO_DCN': w_IO_DCN_val}}}
#     PS_save_name = 'gCal_'+str(parameters_value['no_Plasticity']['Uncoupled']['gCal'])+'_w_IO_DCN_'+str(parameters_value['no_Plasticity']['Uncoupled']['w_IO_DCN']+'_PC_DCN_val_'+str(float(PC_DCN_val)))
    PS_save_name = 'gCal_' + str(parameters_value['no_Plasticity']['Uncoupled']['gCal']) + '_w_IO_DCN_' + str(parameters_value['no_Plasticity']['Uncoupled']['w_IO_DCN']) + '_PC_DCN_val_' + str(float(PC_DCN_val)) + '_leak_' + str(float(leak_val)) + '_sigma_OU_' + str(float(sigma_OU_val)) + '_b_OU_' + str(float(b_OU_val))

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
    N_Cells_PF_events = Noise_frozen.N_Cells_PF_events         
    I_recorded = Noise_frozen.I_recorded
    if parameters_value['filtered']:
        if parameters_value["unfiltered"]:
            if f0 == 0:
                I_recorded = Noise_frozen.I_recorded    
            else:
                I_recorded = TimedArray(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}'].T, dt=dt) 
        else:
            I_recorded = TimedArray(Noise_frozen.Noise_filtered[f'filtered_noise_{f0}'].T, dt=dt) 
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
    w_IO_DCN = parameters_value[plasticity]['Uncoupled']['w_IO_DCN']
    w_IO_DCN_Coupled = parameters_value[plasticity]['Coupled']['w_IO_DCN']
    gCal_in = parameters_value[plasticity]['Uncoupled']['gCal']
    gCal_in_Coupled = parameters_value[plasticity]['Coupled']['gCal']
    
    
    ###################################################################
    ###################### Synapses Values ############################
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
    eqs_pc_dcn = f'I_PC_post += 0.00{PC_DCN_val}*nA'
    eqs_pc_dcn_coupled = f'I_PC_post += 0.00{PC_DCN_val}*nA'
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
    IO, IO_Statemon, IO_Spikemon, IO_rate = IO_neurons(IO_record,N_Cells_IO,Noise_frozen,IO_Values,IO_thresh,dt,dt_rec)
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
    IO_Coupled, IO_Statemon_Coupled, IO_Spikemon_Coupled, IO_rate_Coupled = IO_neurons(IO_record,N_Cells_IO,Noise_frozen,IO_Values,IO_thresh,dt,dt_rec)
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
    PC_DCN_Synapse_Coupled = PC_DCN_syn(PC_Coupled,DCN_Coupled,N_Cells_PC,N_Cells_DCN,eqs_pc_dcn_coupled,dt,dt_rec)
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
        
    IO.g_ls += leak_val*mS*cm**-2
    IO.g_ld += leak_val*mS*cm**-2 
    IO.g_la += leak_val*mS*cm**-2
    
    IO_Coupled.g_ls += leak_val*mS*cm**-2
    IO_Coupled.g_ld += leak_val*mS*cm**-2 
    IO_Coupled.g_la += leak_val*mS*cm**-2
    
    gCal_values = rand_params(gCal_in,1,N_Cells_IO,(0.05/N_Cells_IO))*mS/cm**2
    gCal_values_Coupled = rand_params(gCal_in_Coupled,1,N_Cells_IO,(0.05/N_Cells_IO))*mS/cm**2
    
    dcn_intr = 1.2 #rand_params(1.2,1,N_Cells_DCN,(0.05/N_Cells_DCN))
    
    DCN.I_intrinsic = [dcn_intr]*N_Cells_DCN*nA
    DCN_Coupled.I_intrinsic = [dcn_intr]*N_Cells_DCN*nA
    
    IO.g_Ca_l =  gCal_values
    IO_Coupled.g_Ca_l =  gCal_values_Coupled

    sigma_ou = [sigma_OU_val]*N_Cells_IO
    sigma_ou_Coupled = [sigma_OU_val]*N_Cells_IO
    b_OU = [b_OU_val]*N_Cells_IO
    b_OU_Coupled = [b_OU_val]*N_Cells_IO
    
    IO.sigma_OU = sigma_ou*uA/cm**2
    IO_Coupled.sigma_OU = sigma_ou*uA/cm**2

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
    
#     figure(figsize=(8,4))
#     for io_num in range(40):
#         plot(PC_Statemon.t/ms,PC_Statemon.v[io_num]/mV)
#     plt.show()
    
#     figure(figsize=(8,4))
#     for io_num in range(40):
#         plot(PC_Statemon_Coupled.t/ms,PC_Statemon_Coupled.v[io_num]/mV)
#     plt.show()
    
#     figure(figsize=(15,8))
#     for io_num in range(40):
#         plot(IO_Statemon.t/ms,IO_Statemon.Vs[io_num]/mV)
#     plt.show()
    
#     figure(figsize=(15,8))
#     for io_num in range(40):
#         plot(IO_Statemon_Coupled.t/ms,IO_Statemon_Coupled.Vs[io_num]/mV)
#     plt.show()
    
#     IO_Copy_Synapse_Targets = Synaps.IO_Copy_Synapse_Targets
#     Noise_PC_Synapse_Sources = Synaps.Noise_PC_Synapse_Sources
#     Noise_PC_Synapse_Targets = Synaps.Noise_PC_Synapse_Targets 
#     Noise_gain = noise_gain
#     if plasticity == 'after_Plasticity':
#         net_name = str(N_Cells_PC)+'PC_'+str(N_Cells_DCN)+'DCN_'+str(N_Cells_IO)+'IO'
#         for coupling in ['','_Coupled']:
#             savgol_filter_mon_N_PC_new_weight = output_load_run(PS_save_name,'mon_N_PC',coupling,seed_number,'Plasticity',zebrin,noise_gain,exp_run,net_name,path_data)['new_weight']
#             for col_syn_fil in range(0,len(savgol_filter_mon_N_PC_new_weight)):
#                 t = linspace(0,int(exp_run/second),len(savgol_filter_mon_N_PC_new_weight[col_syn_fil])) 
#                 signala = savgol_filter_mon_N_PC_new_weight[col_syn_fil] # with frequency of 100
#                 output_savgol = signal.savgol_filter(signala, 100, 2)
#                 savgol_filter_mon_N_PC_new_weight[col_syn_fil] = output_savgol
#             if coupling == '_Coupled':
#                 Noise_PC_Synapse_Weights_Coupled = savgol_filter_mon_N_PC_new_weight[:,-1]
#             else:
#                 Noise_PC_Synapse_Weights = savgol_filter_mon_N_PC_new_weight[:,-1]
        
# #         Noise_PC_Synapse_Weights = output_load_run('mon_N_PC','',seed_number,'Plasticity',zebrin,exp_run,net_name,path_data)['new_weight'][:,-1]
# #         Noise_PC_Synapse_Weights_Coupled = output_load_run('mon_N_PC','_Coupled',seed_number,'Plasticity',zebrin,exp_run,net_name,path_data)['new_weight'][:,-1]
        
#     else:
#         Noise_PC_Synapse_Weights = Synaps.Noise_PC_Synapse_Weights
#     PC_DCN_Synapse_Sources = Synaps.PC_DCN_Synapse_Sources
#     PC_DCN_Synapse_Targets = Synaps.PC_DCN_Synapse_Targets
#     DCN_IO_Synapse_Sources = Synaps.DCN_IO_Synapse_Sources 
#     DCN_IO_Synapse_Targets = Synaps.DCN_IO_Synapse_Targets 
#     IO_PC_Synapse_Sources = Synaps.IO_PC_Synapse_Sources
#     IO_PC_Synapse_Targets = Synaps.IO_PC_Synapse_Targets    
#     ###################################################################
#     ###################################################################
#     ############################## CELLS ##############################
#     ###################################################################
#     ###################################################################
#     ############################## NOISE ##############################
#     ###################################################################
#     Noise, Noise_statemon = Noise_neuron(Noise_record,N_Noise,I_recorded,noise_gain,period,dt,dt_rec)
#     ###################################################################
#     ########################## PURKINJE CELL ##########################
#     ###################################################################
#     PC, PC_Statemon, PC_Spikemon, PC_rate = PC_neurons(PC_record,N_Cells_PC,PC_Values,dt,dt_rec)
#     ###################################################################
#     ################## DEEP CEREBELLAR NUCLEI CELLS ###################
#     ###################################################################
#     DCN, DCN_Statemon, DCN_Spikemon, DCN_rate = DCN_neurons(DCN_record,N_Cells_DCN,DCN_Values,dt,dt_rec)
#     ###################################################################
#     ############################# IO ##################################
#     ###################################################################
#     IO, IO_Statemon, IO_Spikemon, IO_rate = IO_neurons(IO_record,N_Cells_IO,IO_Values,IO_thresh,dt,dt_rec)
#     ###################################################################
#     ############################# PF ##################################
#     ###################################################################
#     Input_presyn, Input_presyn_statemon = presyn_inp(Input_presyn_record,I_recorded,plasticity,N_Noise, dt, dt_rec)
#     ###################################################################
#     ############################# Rate ################################
#     ###################################################################
#     syn = rate_meas_func(rate_meas,PC,dt)
#     syn.connect(j='i')  
#     syn.subtract.delay = rate_meas_out  # delay the subtraction
#     ###################################################################
#     ############################# Copy ################################
#     ###################################################################
#     conn_N_PC, mon_N_PC = conn_N_PC_func(conn_N_PC_record,plasticity,N_Copy, Noise_PC_Synapse_Weights, dt, dt_rec)
#     ###################################################################
#     ###################################################################
#     ############################## CELLS COUPLED ######################
#     ###################################################################
#     ###################################################################
#     ############################## NOISE COUPLED ######################
#     ###################################################################
#     Noise_Coupled, Noise_statemon_Coupled = Noise_neuron(Noise_record,N_Noise,I_recorded,noise_gain,period,dt,dt_rec)
#     ###################################################################
#     ########################## PURKINJE CELL COUPLED ##################
#     ###################################################################
#     PC_Coupled, PC_Statemon_Coupled, PC_Spikemon_Coupled, PC_rate_Coupled = PC_neurons(PC_record,N_Cells_PC,PC_Values,dt,dt_rec)
#     ###################################################################
#     ################ DEEP CEREBELLAR NUCLEI CELLS COUPLED #############
#     ###################################################################
#     DCN_Coupled, DCN_Statemon_Coupled, DCN_Spikemon_Coupled, DCN_rate_Coupled = DCN_neurons(DCN_record,N_Cells_DCN,DCN_Values,dt,dt_rec)
#     ###################################################################
#     ############################# IO COUPLED ##########################
#     ###################################################################
#     IO_Coupled, IO_Statemon_Coupled, IO_Spikemon_Coupled, IO_rate_Coupled = IO_neurons(IO_record,N_Cells_IO,IO_Values,IO_thresh,dt,dt_rec)
#     IO_synapse_Coupled = IO_coup_syn(IO_Coupled,eqs_IO_syn) # create synaptic equations and apply full synaptic strength for second network
#     IO_synapse_Coupled.connect() # connect second network
#     ###################################################################
#     ############################# PF COUPLED ##########################
#     ###################################################################
#     Input_presyn_Coupled, Input_presyn_statemon_Coupled = presyn_inp(Input_presyn_record,I_recorded,plasticity,N_Noise, dt, dt_rec)
#     ###################################################################
#     ############################# Rate COUPLED ########################
#     ###################################################################
#     syn_Coupled = rate_meas_func(rate_meas,PC_Coupled,dt)
#     syn_Coupled.connect(j='i')  
#     syn_Coupled.subtract.delay = rate_meas_out  # delay the subtraction
#     ###################################################################
#     ############################# Copy COUPLED ########################
#     ###################################################################
#     conn_N_PC_Coupled, mon_N_PC_Coupled = conn_N_PC_func(conn_N_PC_record,plasticity,N_Copy, Noise_PC_Synapse_Weights, dt, dt_rec)
#     ###################################################################
#     ###################################################################
#     ########################## SYNAPSES ###############################
#     ###################################################################
#     ###################################################################
#     ########################## PC DCN Synapse #########################
#     ###################################################################
#     PC_DCN_Synapse = PC_DCN_syn(PC,DCN,N_Cells_PC,N_Cells_DCN,dt,dt_rec)
#     PC_DCN_Synapse.connect(i=PC_DCN_Synapse_Sources,j=PC_DCN_Synapse_Targets)
#     ###################################################################
#     ########################## DCN IO Synapse #########################
#     ###################################################################
#     DCN_IO_Synapse = DCN_IO_syn(DCN,IO,N_Cells_DCN,N_Cells_IO,w_IO_DCN_Coupled,dt,dt_rec)
#     DCN_IO_Synapse.connect(i=DCN_IO_Synapse_Sources,j=DCN_IO_Synapse_Targets)
#     ###################################################################
#     ########################## IO ConnPC Synapse ######################
#     ###################################################################
#     S_IO_N = Synapses(IO, conn_N_PC, on_pre = 'delta_weight_CS += -0.02*rho_PF_post/Hz', method='euler',dt=dt)  # where f is 
#     S_IO_N.connect(i=IO_PC_Synapse_Sources, j=IO_Copy_Synapse_Targets)
#     ###################################################################
#     ########################### IO PC Synapse #########################
#     ###################################################################
#     IO_PC_Synapse = Synapses(IO, PC, on_pre = eqs_syn_IO_PC_pre, delay=2*ms,method = 'euler',dt=dt)
#     IO_PC_Synapse.connect(i=IO_PC_Synapse_Sources,j=IO_PC_Synapse_Targets)
#     ###################################################################
#     ######################### ConnPC PC Synapse #######################
#     ###################################################################
#     S_N_PC = Synapses(conn_N_PC,PC, eqs_syn_bcm_s_n_pc, method='euler',dt=dt)
#     S_N_PC.connect(i=N_Copy_order, j = Noise_PC_Synapse_Targets)
#     ###################################################################
#     ########################## ConnPC Noise Synapse ###################
#     ###################################################################
#     S_PC_N = Synapses(conn_N_PC,Noise, 'weight_post = new_weight_pre : 1 (summed)', method='euler',dt=dt)
#     S_PC_N.connect(i=N_Copy_order, j = Noise_PC_Synapse_Sources)
#     ###################################################################
#     ############################# Copy rate ###########################
#     ###################################################################
#     copy_rate = Synapses(Input_presyn, conn_N_PC, 'rho_PF_post = rho_presyn_pre : Hz (summed)', method='euler',dt=dt)
#     copy_rate.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
#     ###################################################################
#     ############################ Copy Noise ###########################
#     ###################################################################
#     copy_noise = Synapses(Noise, conn_N_PC, 'I_post = I_pre : amp (summed)', method='euler', dt=dt)
#     copy_noise.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
#     copy_noise.weight = Noise_PC_Synapse_Weights
#     ###################################################################
#     ########################## PC Rate Synapse ########################
#     ###################################################################
#     S_PC_rate = Synapses(PC,conn_N_PC, 'rho_PC_post = New_recent_rate_pre : Hz (summed)', method='euler',dt=dt)
#     S_PC_rate.connect(i=Noise_PC_Synapse_Targets, j =N_Copy_order)
#     ###################################################################
#     ###################################################################
#     ####################### SYNAPSES COUPLED ##########################
#     ###################################################################
#     ###################################################################
#     ########################## PC DCN Synapse #########################
#     ###################################################################
#     PC_DCN_Synapse_Coupled = PC_DCN_syn(PC_Coupled,DCN_Coupled,N_Cells_PC,N_Cells_DCN,dt,dt_rec)
#     PC_DCN_Synapse_Coupled.connect(i=PC_DCN_Synapse_Sources,j=PC_DCN_Synapse_Targets)
#     ###################################################################
#     #################### DCN IO Synapse COUPLED #######################
#     ###################################################################
#     DCN_IO_Synapse_Coupled = DCN_IO_syn(DCN_Coupled,IO_Coupled,N_Cells_DCN,N_Cells_IO,w_IO_DCN_Coupled,dt,dt_rec)
#     DCN_IO_Synapse_Coupled.connect(i=DCN_IO_Synapse_Sources,j=DCN_IO_Synapse_Targets)
#     ###################################################################
#     ################# IO ConnPC Synapse COUPLED #######################
#     ###################################################################
#     S_IO_N_Coupled = Synapses(IO_Coupled, conn_N_PC_Coupled, on_pre = 'delta_weight_CS += -0.02*rho_PF_post/Hz', method='euler',dt=dt)  # where f is 
#     S_IO_N_Coupled.connect(i=IO_PC_Synapse_Sources, j=IO_Copy_Synapse_Targets)
#     ###################################################################
#     ##################### IO PC Synapse COUPLED #######################
#     ###################################################################
#     IO_PC_Synapse_Coupled = Synapses(IO_Coupled, PC_Coupled, on_pre =eqs_syn_IO_PC_pre, delay=2*ms,method = 'euler',dt=dt)
#     IO_PC_Synapse_Coupled.connect(i=IO_PC_Synapse_Sources,j=IO_PC_Synapse_Targets)
#     ###################################################################
#     ################# ConnPC PC Synapse COUPLED #######################
#     ###################################################################
#     S_N_PC_Coupled = Synapses(conn_N_PC_Coupled,PC_Coupled, eqs_syn_bcm_s_n_pc, method='euler',dt=dt)
#     S_N_PC_Coupled.connect(i=N_Copy_order, j = Noise_PC_Synapse_Targets)
#     ###################################################################
#     ############## ConnPC Noise Synapse COUPLED #######################
#     ###################################################################
#     S_PC_N_Coupled = Synapses(conn_N_PC_Coupled,Noise_Coupled, 'weight_post = new_weight_pre : 1 (summed)', method='euler',dt=dt)
#     S_PC_N_Coupled.connect(i=N_Copy_order, j = Noise_PC_Synapse_Sources)
#     ###################################################################
#     ######################### Copy rate COUPLED #######################
#     ###################################################################
#     copy_rate_Coupled = Synapses(Input_presyn_Coupled, conn_N_PC_Coupled, 'rho_PF_post = rho_presyn_pre : Hz (summed)', method='euler',dt=dt)
#     copy_rate_Coupled.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
#     ###################################################################
#     ######################## Copy Noise COUPLED #######################
#     ###################################################################
#     copy_noise_Coupled = Synapses(Noise_Coupled, conn_N_PC_Coupled, 'I_post = I_pre : amp (summed)', method='euler', dt=dt)
#     copy_noise_Coupled.connect(i = Noise_PC_Synapse_Sources, j=N_Copy_order)
#     copy_noise_Coupled.weight = Noise_PC_Synapse_Weights
#     ###################################################################
#     ################### PC Rate Synapse COUPLED #######################
#     ###################################################################
#     S_PC_rate_Coupled = Synapses(PC_Coupled,conn_N_PC_Coupled, 'rho_PC_post = New_recent_rate_pre : Hz (summed)', method='euler',dt=dt)
#     S_PC_rate_Coupled.connect(i=Noise_PC_Synapse_Targets, j =N_Copy_order)
    

#     ###################################################################
#     if plasticity == 'Plasticity':
#         conn_N_PC.thresh_M = 100*Hz
#         conn_N_PC_Coupled.thresh_M = 100*Hz
    
#     IO.g_Ca_l =  gCal_in*mS/cm**2
#     IO_Coupled.g_Ca_l =  gCal_in_Coupled*mS/cm**2

#     IO.sigma_OU = 0.5*uA/cm**2
#     IO_Coupled.sigma_OU = 0.5*uA/cm**2

#     a_OU = 1.5
#     b_OU = 0.5

#     IO.I_OU = a_OU*uA/cm**2
#     IO.I0_OU = a_OU*uA/cm**2

#     IO_Coupled.I_OU = a_OU*uA/cm**2
#     IO_Coupled.I0_OU = a_OU*uA/cm**2

#     ###################################################################
#     ########################### RUN ###################################
#     ###################################################################
#     time_checkpoint('about to run')
#     run(exp_run) # Report on the simulation

    
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
            
        save_path_PS = save_path_noise +'/PS_Analysis'      
        if os.path.exists(save_path_PS) == False: 
            os.mkdir(save_path_PS)
            print("Successfully created the directory %s " % save_path_PS)
            
        save_path_PS_name = save_path_PS +'/'+PS_save_name      
        if os.path.exists(save_path_PS_name) == False: 
            os.mkdir(save_path_PS_name)
            print("Successfully created the directory %s " % save_path_PS_name)
            
        save_path_fig = save_path_PS_name+"/Figures"
        if os.path.exists(save_path_fig) == False: 
            os.mkdir(save_path_fig) 
            print("Successfully created the directory %s " % save_path_fig)
            
        save_path_data = save_path_PS_name+"/Output"
        if os.path.exists(save_path_data) == False: 
            os.mkdir(save_path_data)
            print("Successfully created the directory %s " % save_path_data)
   

    Create_output(plasticity,zebrin,save_path_data,width,Noise_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Input_presyn_statemon,Noise_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled,Input_presyn_statemon_Coupled)
    

    
    
def Create_output(plasticity,zebrin,save_path,width,Noise_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Input_presyn_statemon,Noise_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled,Input_presyn_statemon_Coupled):
       
    Output_Noise = {}
    for key in Noise_statemon.recorded_variables.keys():
        Output_Noise[key] = getattr(Noise_statemon, key)

    Output_PC = {}
    for key in PC_Statemon.recorded_variables.keys():
        Output_PC[key] = getattr(PC_Statemon, key)
    Output_PC['Spikemon'] = PC_Spikemon.t/ms
    Output_PC_Spikes = {}
    PC_Spikemon_Cells = [[]]*PC_Spikemon.values('t').__len__()
    for PC_spike in range(0,PC_Spikemon.values('t').__len__()): 
        Output_PC_Spikes[f'{PC_spike}'] = PC_Spikemon.values('t')[PC_spike]
#     Output_PC['Spikemon_Cells'] = PC_Spikemon_Cells
    Output_PC['Rate'] = PC_rate.rate/Hz
    Output_PC['Rate_time'] = PC_rate.t/ms
    
    
    Output_DCN = {}
    for key in DCN_Statemon.recorded_variables.keys():
        Output_DCN[key] = getattr(DCN_Statemon, key)
    Output_DCN['Spikemon'] = DCN_Spikemon.t/ms
    Output_DCN_Spikes = {}
    DCN_Spikemon_Cells = [[]]*DCN_Spikemon.values('t').__len__()
    for DCN_spike in range(DCN_Spikemon.values('t').__len__()): 
        Output_DCN_Spikes[f'{DCN_spike}'] = DCN_Spikemon.values('t')[DCN_spike]
#     Output_DCN['Spikemon_Cells'] = DCN_Spikemon_Cells
    Output_DCN['Rate'] = DCN_rate.rate/Hz

    Output_IO = {}
    for key in IO_Statemon.recorded_variables.keys():
        Output_IO[key] = getattr(IO_Statemon, key)
    Output_IO['Rate'] = IO_rate.rate/Hz
    Output_IO_Spikes = {}
    IO_Spikemon_Cells = [[]]*IO_Spikemon.values('t').__len__()
    for IO_spike in range(IO_Spikemon.values('t').__len__()): 
        Output_IO_Spikes[f'{IO_spike}'] = IO_Spikemon.values('t')[IO_spike]
#     Output_IO['Spikemon_Cells'] = IO_Spikemon_Cells
    
    if mon_N_PC:
        Output_mon_N_PC = {}
        for key in mon_N_PC.recorded_variables.keys():
            Output_mon_N_PC[key] = getattr(mon_N_PC, key)

    Output_Input_presyn = {}
    for key in Input_presyn_statemon.recorded_variables.keys():
        Output_Input_presyn[key] = getattr(Input_presyn_statemon, key)

    Output_Noise_Coupled = {}
    for key in Noise_statemon_Coupled.recorded_variables.keys():
        Output_Noise_Coupled[key] = getattr(Noise_statemon_Coupled, key)


    Output_PC_Coupled = {}
    for key in PC_Statemon_Coupled.recorded_variables.keys():
        Output_PC_Coupled[key] = getattr(PC_Statemon_Coupled, key)
    Output_PC_Coupled['Spikemon'] = PC_Spikemon_Coupled.t/ms
    Output_PC_Spikes_Coupled = {}
    PC_Spikemon_Cells_Coupled = [[]]*PC_Spikemon_Coupled.values('t').__len__()
    for PC_spike in range(0,PC_Spikemon_Coupled.values('t').__len__()): 
        Output_PC_Spikes_Coupled[f'{PC_spike}'] = PC_Spikemon_Coupled.values('t')[PC_spike]
#     Output_PC_Coupled['Spikemon_Cells'] = PC_Spikemon_Cells_Coupled
    Output_PC_Coupled['Rate'] = PC_rate_Coupled.rate/Hz
    Output_PC_Coupled['Rate_time'] = PC_rate_Coupled.t/ms

    Output_DCN_Coupled = {}
    for key in DCN_Statemon_Coupled.recorded_variables.keys():
        Output_DCN_Coupled[key] = getattr(DCN_Statemon_Coupled, key)
    Output_DCN_Coupled['Spikemon'] = DCN_Spikemon_Coupled.t/ms
    Output_DCN_Spikes_Coupled = {}
    DCN_Spikemon_Cells_Coupled = [[]]*DCN_Spikemon_Coupled.values('t').__len__()
    for DCN_spike in range(DCN_Spikemon.values('t').__len__()): 
        Output_DCN_Spikes_Coupled[f'{DCN_spike}'] = DCN_Spikemon_Coupled.values('t')[DCN_spike]
#     Output_DCN_Coupled['Spikemon_Cells'] = DCN_Spikemon_Cells_Coupled
    Output_DCN_Coupled['Rate'] = DCN_rate_Coupled.rate/Hz

    Output_IO_Coupled = {}
    for key in IO_Statemon_Coupled.recorded_variables.keys():
        Output_IO_Coupled[key] = getattr(IO_Statemon_Coupled, key)
    Output_IO_Coupled['Spikemon'] = IO_Spikemon_Coupled.t/ms
    Output_IO_Spikes_Coupled = {}
    IO_Spikemon_Cells_Coupled = [[]]*IO_Spikemon_Coupled.values('t').__len__()
    for IO_spike in range(IO_Spikemon_Coupled.values('t').__len__()): 
        Output_IO_Spikes_Coupled[f'{IO_spike}'] = IO_Spikemon_Coupled.values('t')[IO_spike]
#     Output_IO_Coupled['Spikemon_Cells'] = IO_Spikemon_Cells_Coupled
    Output_IO_Coupled['Rate'] = IO_rate_Coupled.rate/Hz

    if mon_N_PC_Coupled:
        Output_mon_N_PC_Coupled = {}
        for key in mon_N_PC_Coupled.recorded_variables.keys():
            Output_mon_N_PC_Coupled[key] = getattr(mon_N_PC_Coupled, key)

    Output_Input_presyn_Coupled = {}
    for key in Input_presyn_statemon_Coupled.recorded_variables.keys():
        Output_Input_presyn_Coupled[key] = getattr(Input_presyn_statemon_Coupled, key)


    sio.savemat(os.path.join(save_path, 'Output_Noise.mat'), Output_Noise) 
    sio.savemat(os.path.join(save_path, 'Output_PC.mat'), Output_PC) 
    sio.savemat(os.path.join(save_path, 'Output_PC_spikes.mat'), Output_PC_Spikes) 
    sio.savemat(os.path.join(save_path, 'Output_DCN.mat'), Output_DCN) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_spikes.mat'), Output_DCN_Spikes) 
    sio.savemat(os.path.join(save_path, 'Output_IO.mat'), Output_IO)  
    sio.savemat(os.path.join(save_path, 'Output_IO_spikes.mat'), Output_IO_Spikes) 
    sio.savemat(os.path.join(save_path, 'Output_Input_presyn.mat'), Output_Input_presyn)  
    if mon_N_PC:
        sio.savemat(os.path.join(save_path, 'Output_mon_N_PC.mat'), Output_mon_N_PC) 
    sio.savemat(os.path.join(save_path, 'Output_Noise_Coupled.mat'), Output_Noise_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_PC_Coupled.mat'), Output_PC_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_PC_spikes_Coupled.mat'), Output_PC_Spikes_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_Coupled.mat'), Output_DCN_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_spikes_Coupled.mat'), Output_DCN_Spikes_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_IO_Coupled.mat'), Output_IO_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_IO_spikes_Coupled.mat'), Output_IO_Spikes_Coupled) 
    if mon_N_PC:
        sio.savemat(os.path.join(save_path, 'Output_mon_N_PC_Coupled.mat'), Output_mon_N_PC_Coupled) 

    sio.savemat(os.path.join(save_path, 'Output_Input_presyn_Coupled.mat'), Output_Input_presyn_Coupled) 
    

def output_load_run(PS_save_name,Cell_Name,name,seed_number,plasticity,zebrin,noise_gain,exp_run,net_name,path_data):
    Output = {}

    tuning_range = {}

    net_path = path_data+'Frozen/Networks/'+net_name
    tun_path = path_data+'Simulations/Networks/'
    net_path_tun = tun_path+net_name
    seed_path_tun = net_path_tun+'/Seed_'+str(seed_number)
    run_path_tun = seed_path_tun+'/'+str(int(exp_run/msecond))+'ms'
    plasticity_path_tun = run_path_tun+'/'+plasticity
    zebrin_path_tun = plasticity_path_tun+'/Zebrin_'+zebrin
    noise_gain_path_tun = zebrin_path_tun + '/Noise_gain_'+str(int(noise_gain*10))
    PS_path_tun = noise_gain_path_tun + '/PS_Analysis/' + PS_save_name
    Output = sio.loadmat(PS_path_tun+'/Output/Output_'+Cell_Name+name+'.mat', squeeze_me=True)
    return Output    
    
  
    
    

    
    

        
        