from General_functions import *

def Create_output(plasticity,zebrin,save_path,width,Noise_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Input_presyn_statemon,Noise_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled,Input_presyn_statemon_Coupled):
    time_start = time.monotonic()
    def time_checkpoint(name):
        nonlocal time_start
        e = time.monotonic()
        t = e - time_start
        time_start = e
        print(f'TIME: {name} took {t:.3f}s')
    time_checkpoint('starting save')
    
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
    
       
#     Output_Noise = {}
#     for key in Noise_statemon.recorded_variables.keys():
#         Output_Noise[key] = getattr(Noise_statemon, key)
# #     time_checkpoint('created noise')

#     Output_PC = {}
#     for key in PC_Statemon.recorded_variables.keys():
#         Output_PC[key] = getattr(PC_Statemon, key)
# #         time_checkpoint(key)
#     Output_PC['Spikemon'] = PC_Spikemon.t/ms
# #     time_checkpoint('Spikemon PC')
# #     PC_Spikemon_Cells = [[]]*PC_Spikemon.values('t').__len__()
# #     for PC_spike in range(0,PC_Spikemon.values('t').__len__()): 
# #         PC_Spikemon_Cells[PC_spike] = PC_Spikemon.values('t')[PC_spike]
# #     time_checkpoint('PC_spike PC')
# #     Output_PC['Spikemon_Cells'] = PC_Spikemon_Cells
# #     time_checkpoint('Spikemon_Cells PC')
#     Output_PC['Rate'] = PC_rate.rate/Hz
# #     time_checkpoint('Rate PC')
#     Output_PC['Rate_time'] = PC_rate.t/ms
# #     time_checkpoint('Rate_time PC')
# #     time_checkpoint('created PC')

#     Output_DCN = {}
#     for key in DCN_Statemon.recorded_variables.keys():
#         Output_DCN[key] = getattr(DCN_Statemon, key)
#     Output_DCN['Spikemon'] = DCN_Spikemon.t/ms
#     DCN_Spikemon_Cells = [[]]*DCN_Spikemon.values('t').__len__()
#     for DCN_spike in range(DCN_Spikemon.values('t').__len__()): 
#         DCN_Spikemon_Cells[DCN_spike] = DCN_Spikemon.values('t')[DCN_spike]
#     Output_DCN['Spikemon_Cells'] = DCN_Spikemon_Cells
#     Output_DCN['Rate'] = DCN_rate.rate/Hz

#     Output_IO = {}
#     for key in IO_Statemon.recorded_variables.keys():
#         Output_IO[key] = getattr(IO_Statemon, key)
#     IO_Spikemon_Cells = [[]]*IO_Spikemon.values('t').__len__()
#     for IO_spike in range(IO_Spikemon.values('t').__len__()): 
#         IO_Spikemon_Cells[IO_spike] = IO_Spikemon.values('t')[IO_spike]
#     Output_IO['Spikemon_Cells'] = IO_Spikemon_Cells
#     Output_IO['Rate'] = IO_rate.rate/Hz

#     if mon_N_PC:
#         Output_mon_N_PC = {}
#         for key in mon_N_PC.recorded_variables.keys():
#             Output_mon_N_PC[key] = getattr(mon_N_PC, key)

#     Output_Input_presyn = {}
#     for key in Input_presyn_statemon.recorded_variables.keys():
#         Output_Input_presyn[key] = getattr(Input_presyn_statemon, key)

#     Output_Noise_Coupled = {}
#     for key in Noise_statemon_Coupled.recorded_variables.keys():
#         Output_Noise_Coupled[key] = getattr(Noise_statemon_Coupled, key)


#     Output_PC_Coupled = {}
#     for key in PC_Statemon_Coupled.recorded_variables.keys():
#         Output_PC_Coupled[key] = getattr(PC_Statemon_Coupled, key)
# #     Output_PC_Coupled['Spikemon'] = PC_Spikemon_Coupled.t/ms
# #     PC_Spikemon_Cells_Coupled = [[]]*PC_Spikemon_Coupled.values('t').__len__()
# #     for PC_spike in range(0,PC_Spikemon_Coupled.values('t').__len__()): 
# #         PC_Spikemon_Cells_Coupled[PC_spike] = PC_Spikemon_Coupled.values('t')[PC_spike]
# #     Output_PC_Coupled['Spikemon_Cells'] = PC_Spikemon_Cells_Coupled
#     Output_PC_Coupled['Rate'] = PC_rate_Coupled.rate/Hz
#     Output_PC_Coupled['Rate_time'] = PC_rate_Coupled.t/ms

#     Output_DCN_Coupled = {}
#     for key in DCN_Statemon_Coupled.recorded_variables.keys():
#         Output_DCN_Coupled[key] = getattr(DCN_Statemon_Coupled, key)
#     Output_DCN_Coupled['Spikemon'] = DCN_Spikemon_Coupled.t/ms
#     DCN_Spikemon_Cells_Coupled = [[]]*DCN_Spikemon_Coupled.values('t').__len__()
#     for DCN_spike in range(DCN_Spikemon.values('t').__len__()): 
#         DCN_Spikemon_Cells_Coupled[DCN_spike] = DCN_Spikemon_Coupled.values('t')[DCN_spike]
#     Output_DCN_Coupled['Spikemon_Cells'] = DCN_Spikemon_Cells_Coupled
#     Output_DCN_Coupled['Rate'] = DCN_rate_Coupled.rate/Hz

#     Output_IO_Coupled = {}
#     for key in IO_Statemon_Coupled.recorded_variables.keys():
#         Output_IO_Coupled[key] = getattr(IO_Statemon_Coupled, key)
#     Output_IO_Coupled['Spikemon'] = IO_Spikemon_Coupled.t/ms
#     IO_Spikemon_Cells_Coupled = [[]]*IO_Spikemon_Coupled.values('t').__len__()
#     for IO_spike in range(IO_Spikemon_Coupled.values('t').__len__()): 
#         IO_Spikemon_Cells_Coupled[IO_spike] = IO_Spikemon_Coupled.values('t')[IO_spike]
#     Output_IO_Coupled['Spikemon_Cells'] = IO_Spikemon_Cells_Coupled
#     Output_IO_Coupled['Rate'] = IO_rate_Coupled.rate/Hz

#     if mon_N_PC_Coupled:
#         Output_mon_N_PC_Coupled = {}
#         for key in mon_N_PC_Coupled.recorded_variables.keys():
#             Output_mon_N_PC_Coupled[key] = getattr(mon_N_PC_Coupled, key)

#     Output_Input_presyn_Coupled = {}
#     for key in Input_presyn_statemon_Coupled.recorded_variables.keys():
#         Output_Input_presyn_Coupled[key] = getattr(Input_presyn_statemon_Coupled, key)


#     sio.savemat(os.path.join(save_path, 'Output_Noise.mat'), Output_Noise) 
#     sio.savemat(os.path.join(save_path, 'Output_PC.mat'), Output_PC) 
#     sio.savemat(os.path.join(save_path, 'Output_DCN.mat'), Output_DCN) 
#     sio.savemat(os.path.join(save_path, 'Output_IO.mat'), Output_IO) 
#     sio.savemat(os.path.join(save_path, 'Output_Input_presyn.mat'), Output_Input_presyn)  
#     if mon_N_PC:
#         sio.savemat(os.path.join(save_path, 'Output_mon_N_PC.mat'), Output_mon_N_PC) 
#     sio.savemat(os.path.join(save_path, 'Output_Noise_Coupled.mat'), Output_Noise_Coupled) 
#     sio.savemat(os.path.join(save_path, 'Output_PC_Coupled.mat'), Output_PC_Coupled) 
#     sio.savemat(os.path.join(save_path, 'Output_DCN_Coupled.mat'), Output_DCN_Coupled) 
#     sio.savemat(os.path.join(save_path, 'Output_IO_Coupled.mat'), Output_IO_Coupled) 
#     if mon_N_PC:
#         sio.savemat(os.path.join(save_path, 'Output_mon_N_PC_Coupled.mat'), Output_mon_N_PC_Coupled) 

#     sio.savemat(os.path.join(save_path, 'Output_Input_presyn_Coupled.mat'), Output_Input_presyn_Coupled) 


def output_load_run(Cell_Name,name,seed_number,plasticity,zebrin,noise_gain,exp_run,net_name,path_data,parameters_value,f0):
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
    if parameters_value['filtered']:
        if parameters_value["unfiltered"]:
            if f0 != 0:
                noise_gain_path_tun = noise_gain_path_tun +f'/Noise_filtered_{f0}'  
        else:
            noise_gain_path_tun = noise_gain_path_tun +f'/Noise_filtered_{f0}' 
    Output = sio.loadmat(noise_gain_path_tun+'/Output/Output_'+Cell_Name+name+'.mat', squeeze_me=True)
    return Output