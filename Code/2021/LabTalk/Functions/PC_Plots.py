from General_functions import *

def PC_pop_rate_plot(N_Cells_PC,window,width,time_x,PC_rate,PC_rate_Coupled):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    plot(time_x, PC_rate, label='PC_Uncoupled')
    plot(time_x, PC_rate_Coupled, label='PC_Coupled')
    title('PC Population rate monitor for '+str(N_Cells_PC)+" cells")
    ylabel('Rate [Hz]')
    xlabel('Time [ms]')
    legend()
    show()
    
