from General_functions import *

def DCN_equations():
    eqs_DCN = """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + I_intrinsic - I_PC - w)/C : volt
    dw/dt = (a*(v - EL) - w)/tauw : amp

    dI_PC/dt = (I_PC_max-I_PC)/(30*ms) : amp

    I_intrinsic : amp

    C : farad
    gL : siemens 
    EL : volt
    taum : second
    VT : volt
    DeltaT : volt
    Vcut : volt
    tauw : second
    a : siemens
    b : ampere
    Vr : volt
    tauI : second
    I_PC_max : amp
    """
    return eqs_DCN


def DCN_neurons(N_Cells_DCN,DCN_Values,dt,dt_rec):
    eqs_DCN = DCN_equations()
    
    DCN = NeuronGroup(N_Cells_DCN, model = eqs_DCN, threshold='v>Vcut', reset="v=Vr; w+=b", method='euler', dt=dt)

    for jj in range(0,N_Cells_DCN,1):
        DCN.C[jj] = DCN_Values["DCN_C"].item()[jj]*pF
        DCN.gL[jj] = DCN_Values["DCN_gL"].item()[jj]*nS
        DCN.EL[jj] = DCN_Values["DCN_EL"].item()[jj]*mV
        DCN.VT[jj] = DCN_Values["DCN_VT"].item()[jj]*mV
        DCN.DeltaT[jj] = DCN_Values["DCN_DeltaT"].item()[jj]*mV
        DCN.Vcut[jj] = DCN.VT[jj] + 5*DCN.DeltaT[jj]
        DCN.tauw[jj] = DCN_Values["DCN_tauw"].item()[jj]*ms
        DCN.a[jj] = DCN_Values["DCN_a"].item()[jj]*nS
        DCN.b[jj] = DCN_Values["DCN_b"].item()[jj]*nA
        DCN.Vr[jj] = DCN_Values["DCN_Vr"].item()[jj]*mV
        DCN.I_PC_max[jj] = DCN_Values["DCN_I_PC_max"].item()[jj]*nA
        DCN.v[jj] = DCN_Values["DCN_v"].item()[jj]*mV
        DCN.I_intrinsic[jj] = DCN_Values["DCN_I_intrinsic"].item()[jj]*nA
    DCN.I_PC = [0*nA]*N_Cells_DCN

    DCN_Statemon = StateMonitor(DCN, variables = ['v', 'I_PC','w'], record=True, dt=dt_rec)
    DCN_Spikemon = SpikeMonitor(DCN)
    DCN_rate = PopulationRateMonitor(DCN)

    return DCN, DCN_Statemon, DCN_Spikemon, DCN_rate


def DCN_plot_neurons(N_Cells_DCN, DCN_Statemon, DCN_Spikemon, dt_rec):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Cells_DCN):
        vm = DCN_Statemon[ii].v[:]
        for t in DCN_Spikemon.t:
            i = int(t / dt_rec)
            vm[i] = 20*mV
        plot(DCN_Statemon.t/ms,vm/mV, label='DCN_Cell'+str(ii+1))
    title('Membrane Potential for '+str(N_Cells_DCN)+" cells")
    ylabel('V [mV]')
    xlabel('Time [ms]')
    legend()
    show()
