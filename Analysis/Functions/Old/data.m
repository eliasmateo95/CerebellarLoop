function [Data] = data(Params)
    Data = struct;
    if isfield(Params, 'Coupled' )
        load(Params.Coupled.mat);
        Data.Coupled = struct;
        Data.Coupled.Noise_t = Output.Noise.time;
        Data.Coupled.Noise_I = Output.Noise.I;
        Data.Coupled.PC_v = Output.PC.V;
        Data.Coupled.PC_spikes = Output.PC.spikes;
        Data.Coupled.PC_noise = Output.PC.noise;
        Data.Coupled.PC_firingrate = Output.PC.firingratesmooth;
        Data.Coupled.IO_Vs = Output.IO.Vs;
        Data.Coupled.IO_spikes = Output.IO.spikes;
        Data.Coupled.IO_firingrate = Output.IO.firingratesmooth;
        Data.Coupled.DCN_v = Output.DCN.v;
        Data.Coupled.DCN_spikes = Output.DCN.spikes;
        Data.Coupled.DCN_firingrate = Output.DCN.firingratesmooth;        
    end
    
    
    if isfield(Params, 'Uncoupled' )    
        load(Params.Uncoupled.mat);
        Data.Uncoupled = struct;
        Data.Uncoupled.Noise_t = Output.Noise.time;
        Data.Uncoupled.Noise_I = Output.Noise.I;
        Data.Uncoupled.PC_v = Output.PC.V;
        Data.Uncoupled.PC_spikes = Output.PC.spikes;
        Data.Uncoupled.PC_noise = Output.PC.noise;
        Data.Uncoupled.PC_firingrate = Output.PC.firingratesmooth;
        Data.Uncoupled.IO_Vs = Output.IO.Vs;
        Data.Uncoupled.IO_spikes = Output.IO.spikes;
        Data.Uncoupled.IO_firingrate = Output.IO.firingratesmooth;
        Data.Uncoupled.DCN_v = Output.DCN.v;
        Data.Uncoupled.DCN_spikes = Output.DCN.spikes;
        Data.Uncoupled.DCN_firingrate = Output.DCN.firingratesmooth;
    end
end





