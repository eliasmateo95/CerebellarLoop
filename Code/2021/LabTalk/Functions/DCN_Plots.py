from General_functions import *

def DCN_pop_rate_plot(N_Cells_DCN,window,width,time_x, DCN_rate,DCN_rate_Coupled):
    plt.figure(figsize=(14, 5), dpi= 80, facecolor='w', edgecolor='k')
    plot(time_x, DCN_rate, label='DCN_Uncoupled')
    plot(time_x, DCN_rate_Coupled, label='DCN_Coupled')
    title('DCN Population rate monitor for '+str(N_Cells_DCN)+" cells")
    ylabel('Rate [Hz]')
    xlabel('Time [ms]')
    legend()
    show()
    