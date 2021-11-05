from General_functions import *

def PC_cell_values(N_Cells_PC):
    PC_Values = {}
    PC_Values["PC_C"] = rand_params(75,1,N_Cells_PC,(1.0/N_Cells_PC))  #75*pF  #40 * pF  # 0.77*uF*cm**-2* #1090*pF
    PC_Values["PC_gL"] = rand_params(30,1,N_Cells_PC,(1.0/N_Cells_PC))  #30 * nS
    PC_Values["PC_EL"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #-70.6 * mV
    PC_Values["PC_VT"] = rand_params(-50.4,1,N_Cells_PC,(0.5/N_Cells_PC))  #-50.4 * mV
    PC_Values["PC_DeltaT"] = rand_params(2,1,N_Cells_PC,(0.5/N_Cells_PC))  #2 * mV
    PC_Values["PC_tauw"] = rand_params(144,1,N_Cells_PC,(2.0/N_Cells_PC))  #144*ms
    PC_Values["PC_a"] = rand_params(4,1,N_Cells_PC,(0.5/N_Cells_PC))  #4*nS #2*PC_SingleNeuron.C[jj]/(144*ms) # 
    PC_Values["PC_b"] = rand_params(0.0805,1,N_Cells_PC,(0.001/N_Cells_PC))  #0.0805*nA  #0*nA #
    PC_Values["PC_Vr"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #-70.6*mV
    PC_Values["PC_v"] = rand_params(-70.6,1,N_Cells_PC,(0.5/N_Cells_PC))  #[-70.6*mV]*N_Cells_PC
    PC_Values["PC_I_intrinsic"] = rand_params(1.0,1,N_Cells_PC,(0.2/N_Cells_PC))  #[2*nA]*N_Cells_PC
    
    return PC_Values


def DCN_cell_values(N_Cells_DCN):
    DCN_Values = {}
    DCN_Values["DCN_C"] = rand_params(281,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #281*pF  #40 * pF  # 0.77*uF*cm**-2* #1090*pF
    DCN_Values["DCN_gL"] = rand_params(30,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #30 * nS
    DCN_Values["DCN_EL"] = rand_params(-70.6,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-70.6 * mV
    DCN_Values["DCN_VT"] = rand_params(-50.4,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-50.4 * mV
    DCN_Values["DCN_DeltaT"] = rand_params(2,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #2 * mV
    DCN_Values["DCN_tauw"] = rand_params(30,1,N_Cells_DCN,(1.0/N_Cells_DCN))  #30*ms
    DCN_Values["DCN_a"] = rand_params(4,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #4*nS #2*DCN_SingleNeuron.C[jj]/(144*ms) # 
    DCN_Values["DCN_b"] = rand_params(0.0805,1,N_Cells_DCN,(0.001/N_Cells_DCN))  #0.0805*nA  #0*nA #
    DCN_Values["DCN_Vr"] = rand_params(-65,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #-65*mV
    DCN_Values["DCN_tauI"] = 30*1 #rand_params(30,ms,N_Cells_DCN,(1.0/N_Cells_DCN))  #30*ms
    DCN_Values["DCN_I_PC_max"] = [0*1]*N_Cells_DCN #rand_params(0.1,nA,N_Cells_DCN,(0.009/N_Cells_DCN))  #0*nA
    DCN_Values["DCN_v"] = rand_params(-70.6,1,N_Cells_DCN,(0.5/N_Cells_DCN))  #[-70.6*mV]*N_Cells_DCN
    DCN_Values["DCN_I_intrinsic"] = [2.0*1]*N_Cells_DCN  #rand_params(2.5,nA,N_Cells_DCN,(0.001/N_Cells_DCN))  #[3*nA]*N_Cells_DCN

    return DCN_Values



def IO_cell_values(N_Cells_IO):
    IO_Values = {}
    IO_Values["IO_V_Na"] = rand_params(55,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #55*mvolt
    IO_Values["IO_V_K"] = rand_params(-75,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #-75*mvolt
    IO_Values["IO_V_Ca"] = rand_params(120,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #120*mvolt
    IO_Values["IO_V_l"] = rand_params(10,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #10*mvolt 
    IO_Values["IO_V_h"] = rand_params(-43,1 ,N_Cells_IO,(1.0/N_Cells_IO))  #-43*mvolt 
    IO_Values["IO_Cm"] = rand_params(1,1 ,N_Cells_IO,(0.1/N_Cells_IO))  #1*uF*cm**-2 
    IO_Values["IO_g_Na"] = rand_params(150,1,N_Cells_IO,(1.0/N_Cells_IO))  #150*mS/cm**2
    IO_Values["IO_g_Kdr"] = rand_params(9.0,1,N_Cells_IO,(0.1/N_Cells_IO))  #9.0*mS/cm**2
    IO_Values["IO_g_K_s"] = rand_params(5.0,1,N_Cells_IO,(0.1/N_Cells_IO))  #5.0*mS/cm**2
    IO_Values["IO_g_h"] = rand_params(0.12,1,N_Cells_IO,(0.01/N_Cells_IO))  #0.12*mS/cm**2
    IO_Values["IO_g_Ca_h"] = rand_params(4.5,1,N_Cells_IO,(0.1/N_Cells_IO))  #4.5*mS/cm**2
    IO_Values["IO_g_K_Ca"] = rand_params(35,1,N_Cells_IO,(0.5/N_Cells_IO))  #35*mS/cm**2
    IO_Values["IO_g_Na_a"] = rand_params(240,1,N_Cells_IO,(1.0/N_Cells_IO))  #240*mS/cm**2
    IO_Values["IO_g_K_a"] = rand_params(20,1,N_Cells_IO,(0.5/N_Cells_IO))  #20*mS/cm**2
    IO_Values["IO_g_ls"] = rand_params(0.016,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["IO_g_ld"] = rand_params(0.016,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["IO_g_la"] = rand_params(0.016,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.016*mS/cm**2
    IO_Values["IO_g_int"] = rand_params(0.13,1,N_Cells_IO,(0.001/N_Cells_IO))  #0.13*mS/cm**2
    IO_Values["IO_p"] = rand_params(0.25,1,N_Cells_IO,(0.01/N_Cells_IO))  #0.25
    IO_Values["IO_p2"] = rand_params(0.15,1,N_Cells_IO,(0.01/N_Cells_IO))   #0.15
    IO_Values["IO_g_Ca_l"] =  rand_params(0.375,1,N_Cells_IO,(0.01/N_Cells_IO))   #[0.75*1]*N_Cells_IO

    return IO_Values