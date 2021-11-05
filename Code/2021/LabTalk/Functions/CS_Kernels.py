from General_functions import *


def New_PC(Params,PC):
    PC_new = [[]]*Params.N_Cells_PC
    for ii in range(0,Params.N_Cells_PC):
        vm = PC.v[ii]
#         for t in PC.Spikemon_Cells[ii]:
#             i = int(t / Params.dt_rec)
#             vm[i] = 20*mV
        PC_new[ii] = vm
    return PC_new

def New_PC_learned(Params,PC):
    PC_new = [[]]*Params.N_Cells_PC
    for ii in range(0,Params.N_Cells_PC):
        vm = PC.v[ii]
#         for t in PC.Spikemon_Cells[ii]:
#             i = int(t / Params.dt_rec)
#             vm[i] = 20*mV
        PC_new[ii] = vm
    PC_new_learned = [[]]*len(PC_new)
    for ii in range(0,len(PC_new)):
        PC_new_learned[ii] = PC_new[ii][len(PC_new[ii])//2:]
    return PC_new_learned

def CS_PC_avr_all(Coupled,xx1,avrg_CS_all,avrg_PC_all):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    plot(xx1,avrg_CS_all/mV,label="IO")
    plot(xx1,avrg_PC_all/mV,label="PC")
    title('CS triggered PC averages (all cells) for '+Coupled+' Scenario')
    ylabel('V [mV]')
    xlabel('Time [ms]')
    legend()
    show()
    
def CS_PC_avr(Coupled,Params,xx1,Synapses,avrg_CS,avrg_PC):
    for ii in range(0,Params.N_Cells_PC):
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        plot(xx1,avrg_CS[jj]/mV,label="IO "+str(jj+1))
        plot(xx1,avrg_PC[ii]/mV,label="PC "+str(ii+1))
        title('CS triggered PC averages for '+Coupled+' Scenario')
        ylabel('V [mV]')
        xlabel('Time [ms]')
        legend()
        show()

def CS_SS_avr_all_comp(xx1,avrg_CS_all_No_BCM,avrg_PC_all_No_BCM,avrg_CS_all_Coupled_No_BCM,avrg_PC_all_Coupled_No_BCM,avrg_CS_all_BCM,avrg_PC_all_BCM,avrg_CS_all_Coupled_BCM,avrg_PC_all_Coupled_BCM):
    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle('CS triggered PC averages for all IO Cells (connected to PCs)')
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

    ax1 = plt.subplot(gs1[0])
    ylabel('No BCM \n Complex Spikes []')
    plt.title('Uncoupled')
    ax1.set_xticklabels([])
    ax1 = plt.plot(xx1,avrg_CS_all_No_BCM/mV,label="IO")
    ax1 = plt.plot(xx1,avrg_PC_all_No_BCM/mV,label="PC")

    ax2 = plt.subplot(gs1[1])
    plt.title('Coupled')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2 = plt.plot(xx1,avrg_CS_all_Coupled_No_BCM/mV,label="IO")
    ax2 = plt.plot(xx1,avrg_PC_all_Coupled_No_BCM/mV,label="PC")

    ax3 = plt.subplot(gs1[2])
    ylabel('BCM \n Complex Spikes []')
    xlabel('Time [s]')
    ax3 = plt.plot(xx1,avrg_CS_all_BCM/mV,label="IO")
    ax3 = plt.plot(xx1,avrg_PC_all_BCM/mV,label="PC")

    ax4 = plt.subplot(gs1[3])
    ax4.set_yticklabels([])
    ax4 = plt.plot(xx1,avrg_CS_all_Coupled_BCM/mV,label="IO")
    ax4 = plt.plot(xx1,avrg_PC_all_Coupled_BCM/mV,label="PC")
    xlabel('Time [s]')
    plt.show() 
    
    
def CS_SS_avr_comp(Params,Synapses,xx1,avrg_CS_No_BCM,avrg_PC_No_BCM,avrg_CS_Coupled_No_BCM,avrg_PC_Coupled_No_BCM,avrg_CS_BCM,avrg_PC_BCM,avrg_CS_Coupled_BCM,avrg_PC_Coupled_BCM):
    for ii in range(0,Params.N_Cells_PC):
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        fig.suptitle('CS triggered PC averages for PC Cell '+str(ii+1)+' and IO '+str(jj+1))
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

        ax1 = plt.subplot(gs1[0])
        ylabel('No BCM \n Complex Spikes []')
        plt.title('Uncoupled')
        ax1.set_xticklabels([])
        ax1 = plt.plot(xx1,avrg_CS_No_BCM[jj]/mV,label="IO")
        ax1 = plt.plot(xx1,avrg_PC_No_BCM[ii]/mV,label="PC")

        ax2 = plt.subplot(gs1[1])
        plt.title('Coupled')
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2 = plt.plot(xx1,avrg_CS_Coupled_No_BCM[jj]/mV,label="IO")
        ax2 = plt.plot(xx1,avrg_PC_Coupled_No_BCM[ii]/mV,label="PC")

        ax3 = plt.subplot(gs1[2])
        ylabel('BCM \n Complex Spikes []')
        xlabel('Time [s]')
        ax3 = plt.plot(xx1,avrg_CS_BCM[jj]/mV,label="IO")
        ax3 = plt.plot(xx1,avrg_PC_BCM[ii]/mV,label="PC")

        ax4 = plt.subplot(gs1[3])
        ax4.set_yticklabels([])
        ax4 = plt.plot(xx1,avrg_CS_Coupled_BCM[jj]/mV,label="IO")
        ax4 = plt.plot(xx1,avrg_PC_Coupled_BCM[ii]/mV,label="PC")
        xlabel('Time [s]')
        plt.show()
        