from General_functions import *

def PC_equations():
    eqs_PC = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I_Noise + I_intrinsic  -w)/C : volt
    dw/dt = (a*(v - EL) - w)/(tauw) : amp

    I_intrinsic : amp
    I_Noise  : amp  
    I_Noise_empty : amp
    
    C : farad
    gL : siemens 
    EL : volt
    VT : volt
    DeltaT : volt
    Vcut : volt
    tauw : second
    a : siemens
    b : ampere
    Vr : volt
    
    New_recent_rate = recent_rate+try_new_bcm : Hz
    recent_rate : Hz
    dtry_new_bcm/dt = -try_new_bcm/(50*ms) : Hz
    """
    return eqs_PC


def PC_neurons(N_Cells_PC,PC_Values,dt,dt_rec):
    eqs_PC = PC_equations()
    PC = NeuronGroup(N_Cells_PC, model = eqs_PC, threshold='v>Vcut', reset="v=Vr; w+=b", method='euler', dt=dt)
    
    for jj in range(0,N_Cells_PC,1):
        PC.C[jj] = PC_Values["PC_C"].item()[jj]*pF
        PC.gL[jj] = PC_Values["PC_gL"].item()[jj]*nS
        PC.EL[jj] = PC_Values["PC_EL"].item()[jj]*mV
        PC.VT[jj] = PC_Values["PC_VT"].item()[jj]*mV
        PC.DeltaT[jj] = PC_Values["PC_DeltaT"].item()[jj]*mV
        PC.Vcut[jj] = PC.VT[jj] + 5*PC.DeltaT[jj]
        PC.tauw[jj] = PC_Values["PC_tauw"].item()[jj]*ms
        PC.a[jj] = PC_Values["PC_a"].item()[jj]*nS
        PC.b[jj] = PC_Values["PC_b"].item()[jj]*nA
        PC.Vr[jj] = PC_Values["PC_Vr"].item()[jj]*mV
        PC.I_Noise[jj] = 0.5*nA
        PC.v[jj] = PC_Values["PC_v"].item()[jj]*mV
        PC.I_intrinsic[jj] = PC_Values["PC_I_intrinsic"].item()[jj]*nA

    PC_Statemon = StateMonitor(PC, variables = ['v', 'w', 'I_Noise','I_Noise_empty','I_intrinsic','tauw','recent_rate','New_recent_rate'], record=True, dt=dt_rec)
    PC_Spikemon = SpikeMonitor(PC)
    PC_rate = PopulationRateMonitor(PC)

    return PC, PC_Statemon, PC_Spikemon, PC_rate


def PC_plot_neurons(N_Cells_PC, PC_Statemon, PC_Spikemon, dt_rec):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Cells_PC):
        vm = PC_Statemon[ii].v[:]
        for t in PC_Spikemon.t:
            i = int(t / dt_rec)
            vm[i] = 20*mV
        plot(PC_Statemon.t/ms,vm/mV, label='PC_'+str(ii+1))
    title('Membrane Potential for '+str(N_Cells_PC)+" cells")
    ylabel('V [mV]')
    xlabel('Time [ms]')
    legend()
    show()

def PC_plot_current(N_Cells_PC, N_Noise, PC_Statemon, PC_Spikemon, dt_rec):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Cells_PC):
        plot(PC_Statemon.t/ms,PC_Statemon.I_Noise[ii]/nA, label='PSC_Cell'+str(ii+1))
    title('Post synaptic current for '+str(N_Cells_PC)+" cells and "+str(N_Noise)+" sources")
    ylabel('V [mV]')
    xlabel('Time [ms]')
    legend()
    show()