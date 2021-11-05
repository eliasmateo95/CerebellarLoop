from General_functions import *

def Create_output(Name,width,Noise_statemon,Noise_extended_statemon,PC_Statemon,PC_Spikemon, PC_rate,DCN_Statemon,DCN_Spikemon, DCN_rate,IO_Statemon,IO_Spikemon,IO_rate,mon_N_PC,Noise_statemon_Coupled,Noise_extended_statemon_Coupled,PC_Statemon_Coupled,PC_Spikemon_Coupled, PC_rate_Coupled,DCN_Statemon_Coupled,DCN_Spikemon_Coupled,DCN_rate_Coupled,IO_Statemon_Coupled,IO_Spikemon_Coupled,IO_rate_Coupled,mon_N_PC_Coupled):
    Output_Noise = {}
    Output_Noise['I'] = Noise_statemon.I
    Output_Noise['weight'] = Noise_statemon.weight
    Output_Noise['time'] = Noise_statemon.t/ms
    
    Output_Noise_Extended = {}
    Output_Noise_Extended['I'] = Noise_extended_statemon.I
    Output_Noise_Extended['weight'] = Noise_extended_statemon.weight
    Output_Noise_Extended['time'] = Noise_extended_statemon.t/ms

    Output_PC = {}
    Output_PC['v'] = PC_Statemon.v
    Output_PC['w'] = PC_Statemon.w
    Output_PC['I_Noise'] = PC_Statemon.I_Noise
    Output_PC['I_Noise_empty'] = PC_Statemon.I_Noise_empty
    Output_PC['I_intrinsic'] = PC_Statemon.I_intrinsic
    Output_PC['tauw'] = PC_Statemon.tauw
    Output_PC['recent_rate'] = PC_Statemon.recent_rate
    Output_PC['New_recent_rate'] = PC_Statemon.New_recent_rate
    Output_PC['Spikemon'] = PC_Spikemon.t/ms
    PC_Spikemon_Cells = [[]]*PC_Spikemon.values('t').__len__()
    for ii in range(0,PC_Spikemon.values('t').__len__()):
        PC_Spikemon_Cells[ii] = PC_Spikemon.values('t')[ii]
    Output_PC['Spikemon_Cells'] = PC_Spikemon_Cells
    Output_PC['Rate'] = PC_rate.smooth_rate(window='gaussian', width=width)/Hz
    Output_PC['Rate_time'] = PC_rate.t/ms

    Output_DCN = {}
    Output_DCN['v'] = DCN_Statemon.v
    Output_DCN['I_PC'] = DCN_Statemon.I_PC
    Output_DCN['w'] = DCN_Statemon.w
    Output_DCN['Spikemon'] = DCN_Spikemon.t/ms
    Output_DCN['Rate'] = DCN_rate.smooth_rate(window='gaussian', width=width)/Hz

    Output_IO = {}
    Output_IO['Vs'] = IO_Statemon.Vs
    Output_IO['Vd'] = IO_Statemon.Vd
    Output_IO['Va'] = IO_Statemon.Va
    Output_IO['I_c'] = IO_Statemon.I_c
    Output_IO['Iapp_s'] = IO_Statemon.Iapp_s
    Output_IO['Iapp_d'] = IO_Statemon.Iapp_d
    Output_IO['I_IO_DCN'] = IO_Statemon.I_IO_DCN
    Output_IO['Spikemon'] = IO_Spikemon.t/ms
    IO_Spikemon_Cells = [[]]*IO_Spikemon.values('t').__len__()
    for ii in range(0,IO_Spikemon.values('t').__len__()):
        IO_Spikemon_Cells[ii] = IO_Spikemon.values('t')[ii]
    Output_IO['Spikemon_Cells'] = IO_Spikemon_Cells
    Output_IO['Rate'] = IO_rate.smooth_rate(window='gaussian', width=width)/Hz

    if mon_N_PC:
        Output_mon_N_PC = {}
        if Name == "STDP":
            Output_mon_N_PC['a_PC'] = mon_N_PC.a_PC
            Output_mon_N_PC['a_IO'] = mon_N_PC.a_IO
            Output_mon_N_PC['I'] = mon_N_PC.I
            Output_mon_N_PC['delta_weight'] = mon_N_PC.delta_weight
        elif Name == "_No_BCM":
            Output_mon_N_PC['new_weight'] = mon_N_PC.new_weight
        else:
            Output_mon_N_PC['rho_PF'] = mon_N_PC.rho_PF
            Output_mon_N_PC['rho_PC'] = mon_N_PC.rho_PC
            Output_mon_N_PC['phi'] = mon_N_PC.phi
            Output_mon_N_PC['thresh_M'] = mon_N_PC.thresh_M
            Output_mon_N_PC['delta_weight'] = mon_N_PC.delta_weight
            Output_mon_N_PC['new_weight'] = mon_N_PC.new_weight


    Output_Noise_Coupled = {}
    Output_Noise_Coupled['I'] = Noise_statemon_Coupled.I
    Output_Noise_Coupled['weight'] = Noise_statemon_Coupled.weight
    Output_Noise_Coupled['time'] = Noise_statemon.t/ms
    
    Output_Noise_Extended_Coupled = {}
    Output_Noise_Extended_Coupled['I'] = Noise_extended_statemon_Coupled.I
    Output_Noise_Extended_Coupled['weight'] = Noise_extended_statemon_Coupled.weight
    Output_Noise_Extended_Coupled['time'] = Noise_extended_statemon.t/ms

    Output_PC_Coupled = {}
    Output_PC_Coupled['v'] = PC_Statemon_Coupled.v
    Output_PC_Coupled['w'] = PC_Statemon_Coupled.w
    Output_PC_Coupled['I_Noise'] = PC_Statemon_Coupled.I_Noise
    Output_PC_Coupled['I_Noise_empty'] = PC_Statemon_Coupled.I_Noise_empty
    Output_PC_Coupled['I_intrinsic'] = PC_Statemon_Coupled.I_intrinsic
    Output_PC_Coupled['tauw'] = PC_Statemon_Coupled.tauw
    Output_PC_Coupled['recent_rate'] = PC_Statemon_Coupled.recent_rate
    Output_PC_Coupled['New_recent_rate'] = PC_Statemon_Coupled.New_recent_rate
    Output_PC_Coupled['Spikemon'] = PC_Spikemon_Coupled.t/ms
    PC_Spikemon_Cells_Coupled = [[]]*PC_Spikemon_Coupled.values('t').__len__()
    for ii in range(0,PC_Spikemon_Coupled.values('t').__len__()):
        PC_Spikemon_Cells_Coupled[ii] = PC_Spikemon_Coupled.values('t')[ii]
    Output_PC_Coupled['Spikemon_Cells'] = PC_Spikemon_Cells_Coupled
    Output_PC_Coupled['Rate'] = PC_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz
    Output_PC_Coupled['Rate_time'] = PC_rate_Coupled.t/ms

    Output_DCN_Coupled = {}
    Output_DCN_Coupled['v'] = DCN_Statemon_Coupled.v
    Output_DCN_Coupled['I_PC'] = DCN_Statemon_Coupled.I_PC
    Output_DCN_Coupled['w'] = DCN_Statemon_Coupled.w
    Output_DCN_Coupled['Spikemon'] = DCN_Spikemon_Coupled.t/ms
    Output_DCN_Coupled['Rate'] = DCN_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz

    Output_IO_Coupled = {}
    Output_IO_Coupled['Vs'] = IO_Statemon_Coupled.Vs
    Output_IO_Coupled['Vd'] = IO_Statemon_Coupled.Vd
    Output_IO_Coupled['Va'] = IO_Statemon_Coupled.Va
    Output_IO_Coupled['I_c'] = IO_Statemon_Coupled.I_c
    Output_IO_Coupled['Iapp_s'] = IO_Statemon_Coupled.Iapp_s
    Output_IO_Coupled['Iapp_d'] = IO_Statemon_Coupled.Iapp_d
    Output_IO_Coupled['I_IO_DCN'] = IO_Statemon_Coupled.I_IO_DCN
    Output_IO_Coupled['Spikemon'] = IO_Spikemon_Coupled.t/ms
    IO_Spikemon_Cells_Coupled = [[]]*IO_Spikemon_Coupled.values('t').__len__()
    for ii in range(0,IO_Spikemon_Coupled.values('t').__len__()):
        IO_Spikemon_Cells_Coupled[ii] = IO_Spikemon_Coupled.values('t')[ii]
    Output_IO_Coupled['Spikemon_Cells'] = IO_Spikemon_Cells_Coupled
    Output_IO_Coupled['Rate'] = IO_rate_Coupled.smooth_rate(window='gaussian', width=width)/Hz

    if mon_N_PC_Coupled:
        Output_mon_N_PC_Coupled = {}
        if Name == "STDP":
            Output_mon_N_PC_Coupled['a_PC'] = mon_N_PC.a_PC
            Output_mon_N_PC_Coupled['a_IO'] = mon_N_PC.a_IO
            Output_mon_N_PC_Coupled['I'] = mon_N_PC.I
            Output_mon_N_PC_Coupled['delta_weight'] = mon_N_PC.delta_weight
        elif Name == "_No_BCM":
            Output_mon_N_PC_Coupled['new_weight'] = mon_N_PC_Coupled.new_weight
        else:
            Output_mon_N_PC_Coupled['rho_PF'] = mon_N_PC_Coupled.rho_PF
            Output_mon_N_PC_Coupled['rho_PC'] = mon_N_PC_Coupled.rho_PC
            Output_mon_N_PC_Coupled['phi'] = mon_N_PC_Coupled.phi
            Output_mon_N_PC_Coupled['thresh_M'] = mon_N_PC_Coupled.thresh_M
            Output_mon_N_PC_Coupled['delta_weight'] = mon_N_PC_Coupled.delta_weight
            Output_mon_N_PC_Coupled['new_weight'] = mon_N_PC_Coupled.new_weight

    path = os.getcwd()
    time_now = datetime.datetime.now().strftime("%H:%M")
    save_path = path+"/Data/"+datetime.datetime.now().strftime("%m-%d")+"/"+time_now  
    try:
        os.mkdir(save_path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)    


    sio.savemat(os.path.join(save_path, 'Output_Noise'+str(Name)+'.mat'), Output_Noise) 
    sio.savemat(os.path.join(save_path, 'Output_Noise_Extended'+str(Name)+'.mat'), Output_Noise_Extended)
    sio.savemat(os.path.join(save_path, 'Output_PC'+str(Name)+'.mat'), Output_PC) 
    sio.savemat(os.path.join(save_path, 'Output_DCN'+str(Name)+'.mat'), Output_DCN) 
    sio.savemat(os.path.join(save_path, 'Output_IO'+str(Name)+'.mat'), Output_IO) 
    if mon_N_PC:
        sio.savemat(os.path.join(save_path, 'Output_mon_N_PC'+str(Name)+'.mat'), Output_mon_N_PC) 
    sio.savemat(os.path.join(save_path, 'Output_Noise_Coupled'+str(Name)+'.mat'), Output_Noise_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_Noise_Extended_Coupled'+str(Name)+'.mat'), Output_Noise_Extended_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_PC_Coupled'+str(Name)+'.mat'), Output_PC_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_DCN_Coupled'+str(Name)+'.mat'), Output_DCN_Coupled) 
    sio.savemat(os.path.join(save_path, 'Output_IO_Coupled'+str(Name)+'.mat'), Output_IO_Coupled) 
    if mon_N_PC_Coupled:
        sio.savemat(os.path.join(save_path, 'Output_mon_N_PC_Coupled'+str(Name)+'.mat'), Output_mon_N_PC_Coupled) 

    print(time_now)
    
    
def load_data(Name,time_now,date):
    Output_Noise = sio.loadmat('Data/'+date+'/'+time_now+'/Output_Noise'+'_'+Name+'.mat', squeeze_me=True)
    Output_Noise_Extended = sio.loadmat('Data/'+date+'/'+time_now+'/Output_Noise_Extended'+'_'+Name+'.mat', squeeze_me=True)
    Output_PC = sio.loadmat('Data/'+date+'/'+time_now+'/Output_PC'+'_'+Name+'.mat', squeeze_me=True)
    Output_DCN = sio.loadmat('Data/'+date+'/'+time_now+'/Output_DCN'+'_'+Name+'.mat', squeeze_me=True)
    Output_IO = sio.loadmat('Data/'+date+'/'+time_now+'/Output_IO'+'_'+Name+'.mat', squeeze_me=True)
    try:
        Output_mon_N_PC = sio.loadmat('Data/'+date+'/'+time_now+'/Output_mon_N_PC'+'_'+Name+'.mat', squeeze_me=True)
    except NameError:    
        print("Output_mon_N_PC not defined")
    except:
        print("Something else went wrong")

    Output_Noise_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_Noise_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    Output_Noise_Extended_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_Noise_Extended_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    Output_PC_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_PC_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    Output_DCN_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_DCN_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    Output_IO_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_IO_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    try:
        Output_mon_N_PC_Coupled = sio.loadmat('Data/'+date+'/'+time_now+'/Output_mon_N_PC_Coupled'+'_'+Name+'.mat', squeeze_me=True)
    except NameError:    
        print("Output_mon_N_PC_Coupled not defined")
    except:
        print("Something else went wrong")
        
        
    return Output_Noise,Output_Noise_Extended,Output_PC,Output_DCN,Output_IO,Output_mon_N_PC,Output_Noise_Coupled,Output_Noise_Extended_Coupled,Output_PC_Coupled,Output_DCN_Coupled,Output_IO_Coupled,Output_mon_N_PC_Coupled

def Output_func(Name,Coupled,Output_Noise,Output_Noise_Extended,Output_PC,Output_DCN,Output_IO,Output_mon_N_PC,Output_Noise_Coupled,Output_Noise_Extended_Coupled,Output_PC_Coupled,Output_DCN_Coupled,Output_IO_Coupled,Output_mon_N_PC_Coupled):
    class Noise:
        pass
    class Noise_Extended:
        pass
    class PC:
        pass
    class DCN:
        pass
    class IO:
        pass
    class mon_N_PC:
        pass
    if Coupled == "Coupled":
        Noise.I = Output_Noise_Coupled['I']
        Noise.weight = Output_Noise_Coupled['weight']
        Noise.t = Output_Noise_Coupled['time']

        Noise_Extended.I = Output_Noise_Extended_Coupled["I"]
        Noise_Extended.weight = Output_Noise_Extended_Coupled["weight"]
        Noise_Extended.time = Output_Noise_Extended_Coupled["time"]

        PC.v = Output_PC_Coupled['v']
        PC.w = Output_PC_Coupled['w']
        PC.I_Noise = Output_PC_Coupled['I_Noise']
        PC.I_Noise_empty = Output_PC_Coupled['I_Noise_empty'] 
        PC.I_intrinsic = Output_PC_Coupled['I_intrinsic']
        PC.tauw = Output_PC_Coupled['tauw']
        PC.recent_rate = Output_PC_Coupled['recent_rate']
        PC.New_recent_rate = Output_PC_Coupled['New_recent_rate']
        PC.Spikemon = Output_PC_Coupled['Spikemon']
        PC.rate = Output_PC_Coupled['Rate']
        PC.rate_time = Output_PC_Coupled['Rate_time']
        PC.Spikemon_Cells = Output_PC_Coupled['Spikemon_Cells']

        DCN.v = Output_DCN_Coupled['v']
        DCN.I_PC = Output_DCN_Coupled['I_PC']
        DCN.w = Output_DCN_Coupled['w']
        DCN.Spikemon = Output_DCN_Coupled['Spikemon']
        DCN.rate = Output_DCN_Coupled['Rate']

        IO.Vs = Output_IO_Coupled['Vs']
        IO.Vd = Output_IO_Coupled['Vd']
        IO.Va = Output_IO_Coupled['Va']
        IO.I_c = Output_IO_Coupled['I_c']
        IO.Iapp_s = Output_IO_Coupled['Iapp_s']
        IO.Iapp_d = Output_IO_Coupled['Iapp_d']
        IO.I_IO_DCN = Output_IO_Coupled['I_IO_DCN']
        IO.Spikemon = Output_IO_Coupled['Spikemon']
        IO.Spikemon_Cells = Output_IO_Coupled['Spikemon_Cells']
        IO.rate = Output_IO_Coupled['Rate']

        if Name == "STDP":
            mon_N_PC.a_PC = Output_mon_N_PC_Coupled['a_PC']
            mon_N_PC.a_IO = Output_mon_N_PC_Coupled['a_IO']
            mon_N_PC.I = Output_mon_N_PC_Coupled['I']
            mon_N_PC.delta_weight = Output_mon_N_PC_Coupled['delta_weight']
        elif Name == "No_BCM":
            mon_N_PC.new_weight = Output_mon_N_PC_Coupled['new_weight']
        else:
            mon_N_PC.rho_PF = Output_mon_N_PC_Coupled['rho_PF']
            mon_N_PC.rho_PC = Output_mon_N_PC_Coupled['rho_PC']
            mon_N_PC.phi = Output_mon_N_PC_Coupled['phi']
            mon_N_PC.thresh_M = Output_mon_N_PC_Coupled['thresh_M']
            mon_N_PC.delta_weight = Output_mon_N_PC_Coupled['delta_weight']
            mon_N_PC.new_weight = Output_mon_N_PC_Coupled['new_weight']
    else:

        Noise.I = Output_Noise['I']
        Noise.weight = Output_Noise['weight']
        Noise.t = Output_Noise['time']

        Noise_Extended.I = Output_Noise_Extended["I"]
        Noise_Extended.weight = Output_Noise_Extended["weight"]
        Noise_Extended.time = Output_Noise_Extended["time"]

        PC.v = Output_PC['v']
        PC.w = Output_PC['w']
        PC.I_Noise = Output_PC['I_Noise']
        PC.I_Noise_empty = Output_PC['I_Noise_empty'] 
        PC.I_intrinsic = Output_PC['I_intrinsic']
        PC.tauw = Output_PC['tauw']
        PC.recent_rate = Output_PC['recent_rate']
        PC.New_recent_rate = Output_PC['New_recent_rate']
        PC.Spikemon = Output_PC['Spikemon']
        PC.rate = Output_PC['Rate']
        PC.rate_time = Output_PC['Rate_time']
        PC.Spikemon_Cells = Output_PC['Spikemon_Cells']

        DCN.v = Output_DCN['v']
        DCN.I_PC = Output_DCN['I_PC']
        DCN.w = Output_DCN['w']
        DCN.Spikemon = Output_DCN['Spikemon']
        DCN.rate = Output_DCN['Rate']

        IO.Vs = Output_IO['Vs']
        IO.Vd = Output_IO['Vd']
        IO.Va = Output_IO['Va']
        IO.I_c = Output_IO['I_c']
        IO.Iapp_s = Output_IO['Iapp_s']
        IO.Iapp_d = Output_IO['Iapp_d']
        IO.I_IO_DCN = Output_IO['I_IO_DCN']
        IO.Spikemon = Output_IO['Spikemon']
        IO.Spikemon_Cells = Output_IO['Spikemon_Cells']
        IO.rate = Output_IO['Rate']

        if Name == "STDP":
            mon_N_PC.a_PC = Output_mon_N_PC['a_PC']
            mon_N_PC.a_IO = Output_mon_N_PC['a_IO']
            mon_N_PC.I = Output_mon_N_PC['I']
            mon_N_PC.delta_weight = Output_mon_N_PC['delta_weight']
        elif Name == "No_BCM":
            mon_N_PC.new_weight = Output_mon_N_PC['new_weight']
        else:
            mon_N_PC.rho_PF = Output_mon_N_PC['rho_PF']
            mon_N_PC.rho_PC = Output_mon_N_PC['rho_PC']
            mon_N_PC.phi = Output_mon_N_PC['phi']
            mon_N_PC.thresh_M = Output_mon_N_PC['thresh_M']
            mon_N_PC.delta_weight = Output_mon_N_PC['delta_weight']
            mon_N_PC.new_weight = Output_mon_N_PC['new_weight']
    return Noise, Noise_Extended, PC, DCN, IO, mon_N_PC