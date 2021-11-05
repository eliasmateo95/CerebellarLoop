from General_functions import *

def IO_equations():
    eqs_IO_V = '''
    dVs/dt = (-(I_ds + I_ls + I_Na + I_Ca_l + I_K_dr + I_h + I_as) + Iapp_s + I_OU + I_IO_DCN)/Cm      : volt
    dVd/dt = (-(I_sd + I_ld + I_Ca_h + I_K_Ca + I_c) + Iapp_d)/Cm                    : volt
    dVa/dt = (-(I_K_a + I_sa + I_la + I_Na_a))/Cm                                               : volt
    dI_IO_DCN/dt = (0*uA*cm**-2 - I_IO_DCN)/(30*ms)                                             : amp*meter**-2
    I_c                                                                                         : metre**-2*amp
    Iapp_s                                                                                      : metre**-2*amp
    Iapp_d                                                                                      : metre**-2*amp
    '''
    eqs_IO_Ca = '''
    dCa/dt = (-3*I_Ca_h*((uamp / cm**2)**-1)*mM - 0.075*Ca)/ms                                  : mM
    '''
    eqs_IO_Isom = '''
    I_as    = (g_int/(1-p2))*(Vs-Va)                                                            : metre**-2*amp
    I_ls    = g_ls*(Vs-V_l)                                                                     : metre**-2*amp
    I_ds    = (g_int/p)*(Vs-Vd)                                                                 : metre**-2*amp
    I_Na    = g_Na*m_inf**3*h*(Vs-V_Na)                                                         : metre**-2*amp
    I_Ca_l  = g_Ca_l*k*k*k*l*(Vs-V_Ca)                                                          : metre**-2*amp
    I_K_dr  = g_Kdr*n*n*n*n*(Vs-V_K)                                                            : metre**-2*amp
    I_h     = g_h*q*(Vs-V_h)                                                                    : metre**-2*amp
    I_K_s   = g_K_s*(x_s**4)*(Vs-V_K)                                                           : metre**-2*amp
    '''
    eqs_IO_Iden = '''
    I_sd    = (g_int/(1-p))*(Vd-Vs)                                                             : metre**-2*amp
    I_ld    = g_ld*(Vd-V_l)                                                                     : metre**-2*amp
    I_Ca_h  = g_Ca_h*r*r*(Vd-V_Ca)                                                              : metre**-2*amp
    I_K_Ca  = g_K_Ca*s*(Vd-V_K)                                                                 : metre**-2*amp
    '''
    eqs_IO_Iax = '''
    I_K_a  = g_K_a *x_a**4*(Va-V_K)                                                             : metre**-2*amp
    I_sa   = (g_int/p2)*(Va-Vs)                                                                 : metre**-2*amp
    I_la   = g_la*(Va-V_l)                                                                      : metre**-2*amp
    I_Na_a = g_Na_a*m_a**3*h_a*(Va-V_Na)                                                        : metre**-2*amp
    '''
    eqs_IO_activation = '''
    dh/dt = (h_inf - h)/tau_h                                                                   : 1
    dk/dt = (k_inf - k)/tau_k                                                                   : 1
    dl/dt = (l_inf - l)/tau_l                                                                   : 1
    dn/dt = (n_inf - n)/tau_n                                                                   : 1
    dq/dt = (q_inf - q)/tau_q                                                                   : 1
    dr/dt = (r_inf - r)/tau_r                                                                   : 1
    ds/dt = (s_inf - s)/tau_s                                                                   : 1
    m_a = m_inf_a                                                                               : 1
    dh_a/dt = (h_inf_a - h_a)/tau_h_a                                                           : 1
    dx_a/dt = (x_inf_a - x_a)/tau_x_a                                                           : 1
    dx_s/dt = (x_inf_s - x_s)/tau_x_s                                                           : 1
    '''
    eqs_IO_inf = '''
    m_inf   = alpha_m /(alpha_m+beta_m)                                                         : 1
    h_inf   = alpha_h/(alpha_h+beta_h)                                                          : 1
    k_inf   = 1/(1+e**(-(Vs/mvolt+61.0)/4.2))                                                   : 1
    l_inf   = 1/(1+e**((Vs/mvolt+85.5)/8.5))                                                    : 1
    n_inf   = alpha_n/(alpha_n+beta_n)                                                          : 1
    q_inf   = 1/(1+e**((Vs/mvolt+75.0)/(5.5)))                                                  : 1
    r_inf   = alpha_r/(alpha_r + beta_r)                                                        : 1
    s_inf   = alpha_s/(alpha_s+beta_s)                                                          : 1
    m_inf_a = 1/(1+(e**((-30.0-Va/mvolt)/ 5.5)))                                                : 1
    h_inf_a = 1/(1+(e**((-60.0-Va/mvolt)/-5.8)))                                                : 1
    x_inf_a = alpha_x_a/(alpha_x_a+beta_x_a)                                                    : 1
    x_inf_s = alpha_x_s/(alpha_x_s + beta_x_s)                                                  : 1
    '''
    eqs_IO_tau = '''
    tau_h   = 170.0*msecond/(alpha_h+beta_h)                                                    : second
    tau_k   = 5.0*msecond                                                                       : second
    tau_l   = 1.0*msecond*(35.0+(20.0*e**((Vs/mvolt+160.0)/30.0/(1+e**((Vs/mvolt+84.0)/7.3))))) : second
    tau_n   = 5.0*msecond/(alpha_n+beta_n)                                                      : second
    tau_q   = 1.0*msecond/(e**((-0.086*Vs/mvolt-14.6))+e**((0.07*Vs/mvolt-1.87)))               : second
    tau_r   = 5.0*msecond/(alpha_r + beta_r)                                                    : second
    tau_s   = 1.0*msecond/(alpha_s + beta_s)                                                    : second
    tau_h_a = 1.5*msecond*e**((-40.0-Va/mvolt)/33.0)                                            : second
    tau_x_a = 1.0*msecond/(alpha_x_a + beta_x_a)                                                : second
    tau_x_s = 1.0*msecond/(alpha_x_s + beta_x_s)                                                : second
    '''
    eqs_IO_alpha = '''
    alpha_m   = (0.1*(Vs/mvolt + 41.0))/(1-e**(-(Vs/mvolt+41.0)/10.0))                          : 1
    alpha_h   = 5.0*e**(-(Vs/mvolt+60.0)/15.0)                                                  : 1
    alpha_n   = (Vs/mvolt + 41.0)/(1-e**(-(Vs/mvolt+41.0)/10.0))                                : 1
    alpha_r   = 1.7/(1+e**(-(Vd/mvolt - 5.0)/13.9))                                             : 1
    alpha_s   = ((0.00002*Ca/mM)*int((0.00002*Ca/mM)<0.01) + 0.01*int((0.00002*Ca/mM)>=0.01))   : 1
    alpha_x_a = 0.13*(Va/mvolt + 25.0)/(1-e**(-(Va/mvolt+25.0)/10.0))                           : 1
    alpha_x_s = 0.13*(Vs/mvolt + 25.0)/(1-e**(-(Vs/mvolt+25.0)/10.0))                           : 1
    '''

    eqs_IO_beta = '''
    beta_m = 9.0*e**(-(Vs/mvolt+60.0)/20.0)                                                     : 1
    beta_h = (Vs/mvolt+50.0)/(1-e**(-(Vs/mvolt+50.0)/10.0))                                     : 1
    beta_n = 12.5*e**(-(Vs/mvolt+51.0)/80.0)                                                    : 1
    beta_r = 0.02*(Vd/mvolt + 8.5)/(e**((Vd/mvolt + 8.5)/5.0)-1)                                : 1
    beta_s = 0.015                                                                              : 1
    beta_x_a  = 1.69*e**(-0.0125*(Va/mvolt + 35.0))                                             : 1
    beta_x_s  = 1.69*e**(-0.0125*(Vs/mvolt+ 35.0))                                              : 1
    '''

    eqs_vector = '''
    V_Na                                                                                        : volt
    V_K                                                                                         : volt
    V_Ca                                                                                        : volt
    V_l                                                                                         : volt
    V_h                                                                                         : volt
    Cm                                                                                          : farad*meter**-2
    g_Na                                                                                        : siemens/meter**2
    g_Kdr                                                                                       : siemens/meter**2
    g_Ca_l                                                                                      : siemens/meter**2
    g_h                                                                                         : siemens/meter**2
    g_Ca_h                                                                                      : siemens/meter**2
    g_K_Ca                                                                                      : siemens/meter**2
    g_ls                                                                                        : siemens/meter**2
    g_ld                                                                                        : siemens/meter**2
    g_int                                                                                       : siemens/meter**2
    g_Na_a                                                                                      : siemens/meter**2
    g_K_a                                                                                       : siemens/meter**2
    g_la                                                                                        : siemens/meter**2
    g_K_s                                                                                       : siemens/meter**2
    p                                                                                           : 1
    p2                                                                                          : 1
    '''
    
    eqs_noise = """
    dI_OU/dt = (I0_OU - I_OU)/tau_noise + sigma_OU*xi*tau_noise**-0.5                              : amp*meter**-2 
    I0_OU                                                                                       : amp*meter**-2 
    sigma_OU                                                                                    : amp*meter**-2 
    """
    
    eqs_IO = eqs_IO_beta
    eqs_IO += eqs_IO_alpha
    eqs_IO += eqs_IO_tau
    eqs_IO += eqs_IO_inf
    eqs_IO += eqs_IO_activation
    eqs_IO += eqs_IO_Iax
    eqs_IO += eqs_IO_Iden
    eqs_IO += eqs_IO_Isom
    eqs_IO += eqs_IO_Ca
    eqs_IO += eqs_IO_V
    eqs_IO += eqs_vector
    eqs_IO += eqs_noise
    return eqs_IO

def IO_neurons(N_Cells_IO,IO_Values,thresh,dt,dt_rec):
    eqs_IO = IO_equations()

  ####### Coupled group
    IO = NeuronGroup(N_Cells_IO, model = eqs_IO, threshold=thresh,refractory=thresh, method = 'euler',dt=dt)
    IO_Statemon = StateMonitor(IO, variables = ['Vs','Vd','Va','I_c','Iapp_s','Iapp_d','I_IO_DCN','I_OU'], record = True, dt=dt_rec)
    IO_Spikemon = SpikeMonitor(IO, variables=['Vs'])
    IO_rate = PopulationRateMonitor(IO)

    for ii in range(0, N_Cells_IO, 1):
        IO.V_Na[ii] = IO_Values["IO_V_Na"].item()[ii]*mV
        IO.V_K[ii] = IO_Values["IO_V_K"].item()[ii]*mV
        IO.V_Ca[ii] = IO_Values["IO_V_Ca"].item()[ii]*mV
        IO.V_l[ii] = IO_Values["IO_V_l"].item()[ii]*mV
        IO.V_h[ii] = IO_Values["IO_V_h"].item()[ii]*mV
        IO.Cm[ii] = IO_Values["IO_Cm"].item()[ii]*uF*cm**-2
        IO.g_Na[ii] = IO_Values["IO_g_Na"].item()[ii]*mS/cm**2
        IO.g_Kdr[ii] = IO_Values["IO_g_Kdr"].item()[ii]*mS/cm**2
        IO.g_K_s[ii] = IO_Values["IO_g_K_s"].item()[ii]*mS/cm**2
        IO.g_h[ii] = IO_Values["IO_g_h"].item()[ii]*mS/cm**2
        IO.g_Ca_h[ii] = IO_Values["IO_g_Ca_h"].item()[ii]*mS/cm**2
        IO.g_K_Ca[ii] = IO_Values["IO_g_K_Ca"].item()[ii]*mS/cm**2
        IO.g_Na_a[ii] = IO_Values["IO_g_Na_a"].item()[ii]*mS/cm**2
        IO.g_K_a[ii] = IO_Values["IO_g_K_a"].item()[ii]*mS/cm**2
        IO.g_ls[ii] = IO_Values["IO_g_ls"].item()[ii]*mS/cm**2
        IO.g_ld[ii] = IO_Values["IO_g_ld"].item()[ii]*mS/cm**2
        IO.g_la[ii] = IO_Values["IO_g_la"].item()[ii]*mS/cm**2
        IO.g_int[ii] = IO_Values["IO_g_int"].item()[ii]*mS/cm**2
        IO.p[ii] = IO_Values["IO_p"].item()[ii]
        IO.p2[ii] = IO_Values["IO_p2"].item()[ii]
        IO.g_Ca_l[ii] =  IO_Values["IO_g_Ca_l"].item()[ii]*mS/cm**2

    return IO, IO_Statemon, IO_Spikemon, IO_rate

def IO_syn(IO):
    eqs_IO_syn = ''' I_c_pre = (0.00125*mS/cm**2)*(0.6*e**(-((Vd_pre/mvolt-Vd_post/mvolt)/50)**2) + 0.4)*(Vd_pre-Vd_post) : metre**-2*amp (summed)'''
    IO_synapse = Synapses(IO, IO, eqs_IO_syn)
    return IO_synapse

def IO_coup_syn(IO,eqs_IO_syn):
    IO_synapse = Synapses(IO, IO, eqs_IO_syn)
    return IO_synapse

    
def CS_calculations(N_Cells_IO,IO_Statemon,IO_Spikemon,time_interval,dt_rec):
    IO_spikes_times = []
    IO_spikes_values = []
    for i in range(0,IO_Spikemon.values('t').__len__(),1):
        IO_spikes_times.append(IO_Spikemon.values('t')[i])
        IO_spikes_values.append(IO_Spikemon.values('Vs')[i])

    time_range = []
    time_range_all_spikes = []
    Vs_range = []
    Vs_range_all_spikes =[]
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
            if IO_spikes_times[ii][jj]/ms < time_interval:
                time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range((int((IO_spikes_times[ii][jj]/ms)/(dt_rec/ms))),int((time_interval)/(dt_rec/ms)),1))])
                Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
            elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                diff = ((IO_spikes_times[ii][-1]/ms+time_interval)/(dt_rec/ms)) - size(IO_Statemon.Vs[ii])
                time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int(((time_interval)/(dt_rec/ms))-diff),1))])
                Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
            else:
                time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
                time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                
    return IO_spikes_times, time_range_all_spikes, Vs_range_all_spikes, time_range, Vs_range

