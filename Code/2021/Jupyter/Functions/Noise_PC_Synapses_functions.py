from General_functions import *

def Noise_PC_syn(Noise,PC,N_Noise,N_Cells_PC,dt,dt_rec):
    eqs_syn_Noise_PC_noSTDP = '''
        noise_weight : 1
        I_Noise_post = (noise_weight)*(I_pre)*(1.0/N_Noise) : amp (summed)
    '''
    Noise_PC_Synapse = Synapses(Noise, PC, eqs_syn_Noise_PC_noSTDP,dt=dt)    
    return Noise_PC_Synapse

def Noise_PC_Weights(N_Noise,N_Cells_PC):
    Noise_PC_Synapse_Sources = list(range(0,N_Noise))*N_Cells_PC
    Noise_PC_Synapse_Targets = []
    for pp in range(0,N_Cells_PC):
        Noise_PC_Synapse_Targets += N_Noise * [pp]
    Noise_PC_Synapse_Weights = []
    cc = [0]*N_Cells_PC
    for bb in range(0,N_Cells_PC):
        a1 = []
        for kk in range(0,N_Noise):
            a = uniform(0,N_Noise-cc[bb])
            a1.extend([a])
            cc[bb] += a
        random.shuffle(a1)
        Noise_PC_Synapse_Weights.extend(a1)
    
    return Noise_PC_Synapse_Sources, Noise_PC_Synapse_Targets, Noise_PC_Synapse_Weights 

def Noise_plot_weights(N_Cells_PC, N_Noise, Noise_PC_Synapse_Statemon, dt_rec):
    for ii in range(0,N_Cells_PC):
        plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        for jj in range(ii*N_Noise,(ii+1)*N_Noise):
            plot(Noise_PC_Synapse_Statemon.noise_weight[jj], label="Pre_Syn_Weight_"+str(jj+1)+"_Cell"+str(ii+1))
        title('Post synaptic current for PC cell'+str(N_Cells_PC)+" and "+str(N_Noise)+" sources")
        ylabel('V [mV]')
        xlabel('Time [ms]')
        legend()
        show()
        
        