from General_functions import *

def raster_plots(BCM,learned,step1,Params,Synapses,PC_No_BCM,IO_No_BCM,PC_BCM,IO_BCM,PC_Coupled_No_BCM,IO_Coupled_No_BCM,PC_Coupled_BCM,IO_Coupled_BCM):
    for ii in range(0,Params.N_Cells_PC):
        fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        fig.suptitle('Raster plot for IO Cell '+str(jj+1)+' and PC Cell '+str(ii+1)+" "+BCM) # or plt.suptitle('Main title')
        if BCM == "BCM":
            if learned == "Learned":
                PC_all = PC_BCM.Spikemon_Cells[ii][len(PC_BCM.Spikemon_Cells[ii])//2:]
                if size(IO_BCM.Spikemon_Cells[jj]) < 2:
                    IO_all = [IO_BCM.Spikemon_Cells[jj]]
                else:
                    IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
            else:
                PC_all = PC_BCM.Spikemon_Cells[ii][:len(PC_BCM.Spikemon_Cells[ii])//2]
                if size(IO_BCM.Spikemon_Cells[jj]) < 2:
                    IO_all = [IO_BCM.Spikemon_Cells[jj]]
                else:
                    IO_all = IO_BCM.Spikemon_Cells[jj][:len(IO_BCM.Spikemon_Cells[ii])//2]
        else:
            PC_all = PC_No_BCM.Spikemon_Cells[ii]
            if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
                IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
            else:
                IO_all = IO_No_BCM.Spikemon_Cells[jj]
        ax1 = plt.subplot(gs1[0])
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax1 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax1 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax1 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        plt.title('Uncoupled')
        ylabel('Complex Spikes []')
        xlabel('Time [s]')
        ax2 = plt.subplot(gs1[1])
        ax2.set_yticklabels([])
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        if BCM == "BCM":
            if learned == "Learned":
                PC_all = PC_Coupled_BCM.Spikemon_Cells[ii][len(PC_Coupled_BCM.Spikemon_Cells[ii])//2:]
                if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
                    IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
                else:
                    IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
            else:
                PC_all = PC_Coupled_BCM.Spikemon_Cells[ii][:len(PC_Coupled_BCM.Spikemon_Cells[jj])//2]
                if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
                    IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
                else:
                    IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][:len(IO_Coupled_BCM.Spikemon_Cells[jj])//2]
        else:
            PC_all = PC_Coupled_No_BCM.Spikemon_Cells[ii]
            IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax2 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax2 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax2 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        plt.title('Coupled')
        xlabel('Time [s]')   
        plt.show()

def  raster_plots_both(step1,Params,Synapses,PC_No_BCM,IO_No_BCM,PC_BCM,IO_BCM,PC_Coupled_No_BCM,IO_Coupled_No_BCM,PC_Coupled_BCM,IO_Coupled_BCM):
    for ii in range(0,Params.N_Cells_PC):
        fig = plt.figure(figsize=(14, 10), dpi= 80, facecolor='w', edgecolor='k')
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        fig.suptitle('Raster plot for IO Cell '+str(jj+1)+' and PC Cell '+str(ii+1))
        PC_all = PC_No_BCM.Spikemon_Cells[ii]
        if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_No_BCM.Spikemon_Cells[jj]
        ax1 = plt.subplot(gs1[0])
        ax1.set_xticklabels([])
        learned = ""
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax1 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax1 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax1 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        plt.title('Uncoupled')
        ylabel('No BCM \n Complex Spikes []')
        xlabel('Time [s]')
        ax2 = plt.subplot(gs1[1])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        PC_all = PC_Coupled_No_BCM.Spikemon_Cells[ii]
        IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax2 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax2 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax2 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        xlabel('Time [s]')
        plt.title('Coupled')
        PC_all = PC_BCM.Spikemon_Cells[ii][len(PC_BCM.Spikemon_Cells[ii])//2:]
        if size(IO_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
        ax3 = plt.subplot(gs1[2])
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax3 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax3 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax3 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        ylabel('BCM \n Complex Spikes []')
        xlabel('Time [s]')
        ax4 = plt.subplot(gs1[3])
        ax4.set_yticklabels([])
        PC_all = PC_Coupled_BCM.Spikemon_Cells[ii][len(PC_Coupled_BCM.Spikemon_Cells[ii])//2:]
        if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax4 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
            ax4 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax4 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        xlabel('Time [s]')  
        plt.show() 
        
        
        
        
def hist_rast(step1,Params,Synapses,PC_No_BCM,IO_No_BCM,PC_BCM,IO_BCM,PC_Coupled_No_BCM,IO_Coupled_No_BCM,PC_Coupled_BCM,IO_Coupled_BCM):
    con_spike_times_No_BCM = []
    con_spike_times_Coupled_No_BCM = []
    con_spike_times_BCM = []
    con_spike_times_Coupled_BCM = []
    for ii in range(0,Params.N_Cells_PC):
        fig = plt.figure(figsize=(14, 10), dpi= 80, facecolor='w', edgecolor='k')
        gs1 = gridspec.GridSpec(2, 2)
        gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        PC_all = PC_No_BCM.Spikemon_Cells[ii]
        fig.suptitle('Raster plot for IO Cell '+str(jj+1)+' and PC Cell '+str(ii+1))
        if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_No_BCM.Spikemon_Cells[jj]
        ax1 = plt.subplot(gs1[0])
        ax1.set_xticklabels([])
        learned = ""
        spike_times_No_BCM = []
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax1 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            spike_times_No_BCM = append(spike_times_No_BCM,bb)
            con_spike_times_No_BCM = append(con_spike_times_No_BCM,bb)
            ax1 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax1 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        plt.title('Uncoupled')
        ylabel('No BCM \n Complex Spikes []')
        xlabel('Time [s]')
        ax2 = plt.subplot(gs1[1])
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        PC_all = PC_Coupled_No_BCM.Spikemon_Cells[ii]
        IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        spike_times_Coupled_No_BCM = []
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax2 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            spike_times_Coupled_No_BCM = append(spike_times_Coupled_No_BCM,bb)
            con_spike_times_Coupled_No_BCM = append(con_spike_times_Coupled_No_BCM,bb)
            ax2 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax2 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        xlabel('Time [s]')
        plt.title('Coupled')
        PC_all = PC_BCM.Spikemon_Cells[ii][len(PC_BCM.Spikemon_Cells[ii])//2:]
        if size(IO_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
        ax3 = plt.subplot(gs1[2])
        spike_times_BCM = []
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax3 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            spike_times_BCM = append(spike_times_BCM,bb)
            con_spike_times_BCM = append(con_spike_times_BCM,bb)
            ax3 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax3 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        ylabel('BCM \n Complex Spikes []')
        xlabel('Time [s]')
        ax4 = plt.subplot(gs1[3])
        ax4.set_yticklabels([])
        PC_all = PC_Coupled_BCM.Spikemon_Cells[ii][len(PC_Coupled_BCM.Spikemon_Cells[ii])//2:]
        if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
        spike_times_Coupled_BCM = []
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            bb = []
            for ss in range(0,len(PC_all)):
                if start<=PC_all[ss]<=end:
                    bb.append(PC_all[ss])
                else:
                    continue
            ax4 = plt.scatter([x -IO_all[kk] for x in bb],[kk]*len(bb),color='black',s=0.25)
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            spike_times_Coupled_BCM = append(spike_times_Coupled_BCM,bb)
            con_spike_times_Coupled_BCM = append(con_spike_times_Coupled_BCM,bb)
            ax4 = plt.scatter([x -IO_all[kk] for x in aa],[kk]*len(aa),color='blue',s=0.75)
            ax4 = plt.scatter(IO_all[kk]-IO_all[kk],kk,color='red',s=0.75)
        xlabel('Time [s]')  
        plt.show() 

    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle('Histogram plot for IO Cell '+str(jj+1))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
    ax1 = plt.subplot(gs1[0])
    ylabel('No BCM \n Complex Spikes []')
    plt.title('Uncoupled')
    ax1.set_xticklabels([])
    ax1 = plt.hist(spike_times_No_BCM, bins=150)
    ax2 = plt.subplot(gs1[1])
    plt.title('Coupled')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2 = plt.hist(spike_times_Coupled_No_BCM, bins=150)
    ax3 = plt.subplot(gs1[2])
    ax3 = plt.hist(spike_times_BCM, bins=150)
    ylabel('BCM \n Complex Spikes []')
    xlabel('Time [s]')
    ax4 = plt.subplot(gs1[3])
    ax4.set_yticklabels([])
    ax4 = plt.hist(spike_times_Coupled_BCM, bins=150)
    xlabel('Time [s]')
    plt.show() 



def hist_con(step1,Params,Synapses,IO_No_BCM,IO_BCM,IO_Coupled_No_BCM,IO_Coupled_BCM):
    con_spike_times_No_BCM = []
    con_spike_times_Coupled_No_BCM = []
    con_spike_times_BCM = []
    con_spike_times_Coupled_BCM = []
    for ii in range(0,Params.N_Cells_PC):
        jj = Synapses.IO_PC_Synapse_Sources[ii]
        if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_No_BCM.Spikemon_Cells[jj]
        learned = ""
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            con_spike_times_No_BCM = append(con_spike_times_No_BCM,bb)
        IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            con_spike_times_Coupled_No_BCM = append(con_spike_times_Coupled_No_BCM,bb)
        if size(IO_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            con_spike_times_BCM = append(con_spike_times_BCM,bb)
        if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            con_spike_times_Coupled_BCM = append(con_spike_times_Coupled_BCM,bb) 

    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle('Histogram plot for all IO Cells (connected to PCs)')
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
    ax1 = plt.subplot(gs1[0])
    ylabel('No BCM \n Complex Spikes []')
    plt.title('Uncoupled')
    ax1.set_xticklabels([])
    ax1 = plt.hist(con_spike_times_No_BCM, bins=150)
    ax2 = plt.subplot(gs1[1])
    plt.title('Coupled')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2 = plt.hist(con_spike_times_Coupled_No_BCM, bins=150)
    ax3 = plt.subplot(gs1[2])
    ax3 = plt.hist(con_spike_times_BCM, bins=150)
    ylabel('BCM \n Complex Spikes []')
    xlabel('Time [s]')
    ax4 = plt.subplot(gs1[3])
    ax4.set_yticklabels([])
    ax4 = plt.hist(con_spike_times_Coupled_BCM, bins=150)
    xlabel('Time [s]')
    plt.show() 
    
    
def hist_all(step1,Params,Synapses,IO_No_BCM,IO_BCM,IO_Coupled_No_BCM,IO_Coupled_BCM):
    all_spike_times_No_BCM = []
    all_spike_times_Coupled_No_BCM = []
    all_spike_times_BCM = []
    all_spike_times_Coupled_BCM = []
    for jj in range(0,Params.N_Cells_IO):
        learned = ""
        if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            all_spike_times_No_BCM = append(all_spike_times_No_BCM,bb)
        IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            all_spike_times_Coupled_No_BCM = append(all_spike_times_Coupled_No_BCM,bb)
        if size(IO_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            all_spike_times_BCM = append(all_spike_times_BCM,bb)
        if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            all_spike_times_Coupled_BCM = append(all_spike_times_Coupled_BCM,bb)

    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle('Histogram plot for all IO Cells')
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
    ax1 = plt.subplot(gs1[0])
    ylabel('No BCM \n Complex Spikes []')
    plt.title('Uncoupled')
    ax1.set_xticklabels([])
    ax1 = plt.hist(all_spike_times_No_BCM, bins=150)
    ax2 = plt.subplot(gs1[1])
    plt.title('Coupled')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2 = plt.hist(all_spike_times_Coupled_No_BCM, bins=150)
    ax3 = plt.subplot(gs1[2])
    ax3 = plt.hist(all_spike_times_BCM, bins=150)
    ylabel('BCM \n Complex Spikes []')
    xlabel('Time [s]')
    ax4 = plt.subplot(gs1[3])
    ax4.set_yticklabels([])
    ax4 = plt.hist(all_spike_times_Coupled_BCM, bins=150)
    xlabel('Time [s]')
    plt.show() 
    
def hist_rest(step1,Params,Synapses,IO_No_BCM,IO_BCM,IO_Coupled_No_BCM,IO_Coupled_BCM):
    rest_spike_times_No_BCM = []
    rest_spike_times_Coupled_No_BCM = []
    rest_spike_times_BCM = []
    rest_spike_times_Coupled_BCM = []
    for jj in range(0,Params.N_Cells_IO):
        learned = ""
        if jj in Synapses.IO_PC_Synapse_Sources:
            continue
        if size(IO_No_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_No_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            rest_spike_times_No_BCM = append(rest_spike_times_No_BCM,bb)
        IO_all = IO_Coupled_No_BCM.Spikemon_Cells[jj]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if learned == "Learned":
                if start<int(Params.exp_run/second):
                    continue
            else:
                if start>int(Params.exp_run/second):
                    continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            rest_spike_times_Coupled_No_BCM = append(rest_spike_times_Coupled_No_BCM,bb)
        if size(IO_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_BCM.Spikemon_Cells[jj][len(IO_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            rest_spike_times_BCM = append(rest_spike_times_BCM,bb)
        if size(IO_Coupled_BCM.Spikemon_Cells[jj]) < 2:
            IO_all = [IO_Coupled_BCM.Spikemon_Cells[jj]]
        else:
            IO_all = IO_Coupled_BCM.Spikemon_Cells[jj][len(IO_Coupled_BCM.Spikemon_Cells[jj])//2:]
        for kk in range(0,size(IO_all)):
            start = IO_all[kk]-step1/1000
            end = IO_all[kk]+step1/1000
            if start<int(Params.exp_run/second):
                continue
            aa = []
            bb = []
            for ll in range(0,len(IO_all)):
                if start<=IO_all[ll]<=end:
                    aa.append(IO_all[ll])
                else:
                    continue
                if IO_all[kk]<IO_all[ll]<=end:
                    bb.append(IO_all[ll])
            bb = [x -IO_all[kk] for x in bb]
            rest_spike_times_Coupled_BCM = append(rest_spike_times_Coupled_BCM,bb)

    fig = plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle('Histogram plot for all IO Cells not connected to PCs')
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 
    ax1 = plt.subplot(gs1[0])
    ylabel('No BCM \n Complex Spikes []')
    plt.title('Uncoupled')
    ax1.set_xticklabels([])
    ax1 = plt.hist(rest_spike_times_No_BCM, bins=150)
    ax2 = plt.subplot(gs1[1])
    plt.title('Coupled')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2 = plt.hist(rest_spike_times_Coupled_No_BCM, bins=150)
    ax3 = plt.subplot(gs1[2])
    ax3 = plt.hist(rest_spike_times_BCM, bins=150)
    ylabel('BCM \n Complex Spikes []')
    xlabel('Time [s]')
    ax4 = plt.subplot(gs1[3])
    ax4.set_yticklabels([])
    ax4 = plt.hist(rest_spike_times_Coupled_BCM, bins=150)
    xlabel('Time [s]')
    plt.show() 
