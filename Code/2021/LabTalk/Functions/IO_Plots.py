from General_functions import *

def IO_plot_neurons(N_Cells_IO,IO_Statemon, time, Name):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Cells_IO):
        plot(time,IO_Statemon.Vs[ii]/mV, label='IO_Cell'+str(ii+1))
    title('Membrane Potential for '+str(N_Cells_IO)+" cells " + Name +" Scenario")
    ylabel('V [mV]')
    xlabel('Time [ms]')
    legend()
    show()

def IO_plot_coupling_connectivity(N_Cells_IO,IO_Statemon):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    for ii in range(0,N_Cells_IO):
        plot(IO_Statemon.t/ms,IO_Statemon.I_c[ii]/(uA/cmeter**2), label='I_c_IO_Cell'+str(ii+1))
    title('Coupling Connectivity')
    ylabel('I_c [mS/cm\u00B2]')
    xlabel('Time [ms]')
    legend()
    show()
    
    

def CS_all_spikes(N_Cells_IO,time_range_all_spikes,Vs_range_all_spikes,IO_spikes_times,Name):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    title('CS Kernels for '+str(N_Cells_IO)+" cells " + Name +" Scenario")
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
            plot(time_range_all_spikes[jj],Vs_range_all_spikes[jj])
    ylabel('V [mV]')
    xlabel('Time [ms]')
    show()

def CS_averages_spikes(N_Cells_IO,time_range,Vs_range,Name): 
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    title('CS Triggered averages for '+str(N_Cells_IO)+" cells " + Name +" Scenario")
    plot(time_range[0],np.transpose(Vs_range).mean(axis = 1))
    ylabel('V [mV]')
    xlabel('Time [ms]')
    show()
    
    
def IO_pop_rate_plot(N_Cells_IO,window,width,time_x,IO_rate,IO_rate_Coupled):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    plot(time_x, IO_rate, label='IO_Uncoupled')
    plot(time_x, IO_rate_Coupled, label='IO_Coupled')
    title('Population rate monitor for '+str(N_Cells_IO)+" cells")
    ylabel('Rate [Hz]')
    xlabel('Time [ms]')
    legend()
    show()

# def CS_trig_noise_avrg_sum(N_Noise,IO_PC_Synapse_Sources,IO_Statemon,Noise_PC_Synapse_Statemon,IO_spikes_times,time_interval,dt_rec):
#     for kk in range(0,size(IO_PC_Synapse_Sources)):
#         ii = IO_PC_Synapse_Sources[kk]
#         time_range = []
#         Vs_range = []
#         Noise_average = []
#         Noise_2plot = Noise_PC_Synapse_Statemon.I_Noise[kk*N_Noise]
#         for jj in range(0,size(IO_spikes_times[ii])):
#                 if IO_spikes_times[ii][jj]/ms < time_interval:
#                     time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range((int((IO_spikes_times[ii][jj]/ms)/(dt_rec/ms))),int((time_interval)/(dt_rec/ms)),1))])
#                     Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
#                 elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
#                     diff = ((IO_spikes_times[ii][-1]/ms+time_interval)/(dt_rec/ms)) - size(IO_Statemon.Vs[ii])
#                     time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int(((time_interval)/(dt_rec/ms))-diff),1))])
#                     Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
#                 else:
#                     time_range_all_spikes.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
#                     Vs_range_all_spikes.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range_all_spikes[jj][-1]*ms/dt_rec+1)])/mV)
#                     time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
#                     Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
#                     Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)]/nA)
#         if Vs_range == []:
#             continue
#         else:
#             fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
#             ax = fig.add_subplot(111)
#             lns1 = ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1), label = 'PSC Cell '+str(kk))
#             ax.tick_params(axis='y', labelcolor='tab:blue')
#             ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
#             lns2 = ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = 'IO Cell '+str(IO_PC_Synapse_Sources[kk]), color='tab:red')
#             ax2.tick_params(axis='y', labelcolor='tab:red')
#             lns = lns1+lns2
#             labs = [l.get_label() for l in lns]
#             ax.legend(lns, labs, loc=0)
#             ax.set_xlabel('Time [ms]')
#             ax.set_ylabel('I [nA]', color='tab:blue')
#             ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
#             fig.tight_layout()  # otherwise the right y-label is slightly clipped
#             plt.show()
            
            
            
def CS_noise_conn_plots(N_Cells_PC,N_Noise,Noise_PC_Synapse_Statemon,IO_Statemon,IO_PC_Synapse_Sources):
    for ii in range(0,N_Cells_PC):
        fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        lns1 = ax.plot(Noise_PC_Synapse_Statemon.t/ms,Noise_PC_Synapse_Statemon.I_Noise[ii*N_Noise]/nA, label='PSC Cell '+str(ii+1), color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        lns2 = ax2.plot(IO_Statemon.t/ms,IO_Statemon.Vs[IO_PC_Synapse_Sources[ii]]/mV, label='IO_Cell '+str(IO_PC_Synapse_Sources[ii]), color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('I [nA]', color='tab:blue')
        ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
        
        
def PSC_each_synapse(N_Cells_PC,N_Noise,Noise_PC_Synapse_Statemon):
    PSC_added = []
    for ii in range(0,N_Cells_PC):
        plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        for jj in range(0,N_Noise):
            PSC_added.append(Noise_PC_Synapse_Statemon.noise_weight[ii+jj*N_Cells_PC]*Noise_PC_Synapse_Statemon.I_Noise[ii+jj*N_Cells_PC]/nA)
            plot(Noise_PC_Synapse_Statemon.t/ms,PSC_added[ii*N_Noise+jj], label='PSC Cell '+str(ii+1)+' Noise Source '+str(jj+1))
        title('Post synaptic current for PC cell '+str(ii+1)+" cells and "+str(N_Noise)+" sources")
        ylabel('[-]')
        xlabel('Time [ms]')
        legend()
        show()
        
def CS_trig_noise_avrg_sum(N_Noise,IO_PC_Synapse_Sources,IO_Statemon,Noise_PC_Synapse_Statemon,IO_spikes_times,time_interval,dt_rec):
    for kk in range(0,size(IO_PC_Synapse_Sources)):
        ii = IO_PC_Synapse_Sources[kk]
        time_range = []
        Vs_range = []
        Noise_average = []
        Noise_2plot = Noise_PC_Synapse_Statemon.I_Noise[kk*N_Noise]
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)]/nA)
        if Vs_range == []:
            continue
        else:
            fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            lns1 = ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1), label = 'PSC Cell '+str(kk))
            ax.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            lns2 = ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = 'IO Cell '+str(IO_PC_Synapse_Sources[kk]), color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc=0)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('I [nA]', color='tab:blue')
            ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            
            
def CS_trig_noise_avrg_all_cells(N_Cells_IO,Noise_statemon,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    for ii in range(0,N_Cells_IO):
        time_range = []
        Vs_range = []
        Noise_average = []
        Noise_2plot = np.transpose(Noise_statemon.I).mean(axis = 1)
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)]/nA)
        if Vs_range == []:
            continue
        else:
            fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1))
            ax.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = 'IO Cell '+str(ii+1), color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('I [nA]', color='tab:blue')
            ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            legend()
            plt.show()
            
def CS_trig_noise_avrg(N_Cells_IO,Noise_statemon,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    time_range = []
    Vs_range = []
    Noise_average = []
    Noise_2plot = np.transpose(Noise_statemon.I).mean(axis = 1)
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)]/nA)
    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1), label = str(size(Noise_statemon))+' Noise Sources')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = str(N_Cells_IO)+' IO Cells', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('I [nA]', color='tab:blue')
    ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    legend()
    plt.show()
    
    
def CS_trig_PC_pop_avrg(N_Cells_IO,window,width,PC_rate,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    time_range = []
    Vs_range = []
    Noise_average = []
    Noise_2plot = PC_rate.smooth_rate(window=window, width=width)/Hz
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])
    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1))
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = str(N_Cells_IO)+' IO Cells', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Rate [Hz]', color='tab:blue')
    ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    legend()
    plt.show()
    
    
def CS_trig_DCN_pop_avrg(N_Cells_IO,window,width,DCN_rate,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    time_range = []
    Vs_range = []
    Noise_average = []
    Noise_2plot = DCN_rate.smooth_rate(window=window, width=width)/Hz
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])
    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1))
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = str(N_Cells_IO)+' IO Cells', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Rate [Hz]', color='tab:blue')
    ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    legend()
    plt.show()
    
    
def CS_trig_PSC_avrg(N_Cells_IO,N_Cells_PC,PC_Statemon,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    for kk in range(0,N_Cells_PC):
        time_range = []
        Vs_range = []
        Noise_average = []
        Noise_2plot = PC_Statemon.I_Noise[kk]
        for ii in range(0,N_Cells_IO):
            for jj in range(0,size(IO_spikes_times[ii])):
                    if IO_spikes_times[ii][jj]/ms < time_interval:
                        continue
                    elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                        continue
                    else:
                        time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                        Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                        Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])
        fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        lns1 = ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1)/nA, label = 'PSC_PC_Cell_'+str(kk+1))
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        lns2 = ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        lns = lns1
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Rate [Hz]', color='tab:blue')
        ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()
        
        
        
def CS_trig_PSC_avrg_all(N_Cells_IO,PC_Statemon,IO_Statemon,time_interval,IO_spikes_times,dt_rec):
    time_range = []
    Vs_range = []
    Noise_average = []
    Noise_2plot = np.transpose(PC_Statemon.I_Noise).mean(axis = 1)
    for ii in range(0,N_Cells_IO):
        for jj in range(0,size(IO_spikes_times[ii])):
                if IO_spikes_times[ii][jj]/ms < time_interval:
                    continue
                elif ((IO_spikes_times[ii][jj]/ms+time_interval)/(dt_rec/ms)) > size(IO_Statemon.Vs[ii]): 
                    continue
                else:
                    time_range.append([x * (dt_rec/ms) for x in list(range(-int((time_interval)/(dt_rec/ms)),int((time_interval)/(dt_rec/ms)),1))])
                    Vs_range.append((IO_Statemon.Vs[ii][int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)])/mV)
                    Noise_average.append(Noise_2plot[int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][0]*ms/dt_rec):int(round(IO_spikes_times[ii][jj]/dt_rec))+int(time_range[jj][-1]*ms/dt_rec+1)]/nA)
    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    ax.plot(time_range[0],np.transpose(Noise_average).mean(axis = 1), label = str(size(Noise_statemon))+' Noise Sources')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(time_range[0],np.transpose(Vs_range).mean(axis = 1), label = str(N_Cells_IO)+' IO Cells', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('I [nA]', color='tab:blue')
    ax2.set_ylabel('V [mV]', color='tab:red')  # we already handled the x-label with ax1
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    legend()
    plt.show()
    