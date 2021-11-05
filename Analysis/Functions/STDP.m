function [Data_STDP] = STDP(Data,Params)
Data_STDP = struct;
if isfield(Params, 'Coupled' )
        load(Params.Coupled.mat);
        coldata = size(Data.Coupled.Noise_t,2);
        Data_STDP.Coupled = struct;
        Data_STDP.Coupled.Noise_t = Data.Coupled.Noise_t(:,1:coldata/2);
        Data_STDP.Coupled.Noise_I = Data.Coupled.Noise_I(:,1:coldata/2);
        Data_STDP.Coupled.PC_v = Data.Coupled.PC_v(:,1:coldata/2);
        colPC = size(Data.Coupled.PC_spikes,2);
        for ii = 1:colPC
            col = size(Data.Coupled.PC_spikes{1,ii},2);
            for kk = 1:col
                if Data.Coupled.Noise_t(end)/2000 <Data.Coupled.PC_spikes{1,ii}(kk)
                    break
                end
                Data_STDP.Coupled.PC_spikes{1,ii}(kk) = Data.Coupled.PC_spikes{1,ii}(kk);
            end
        end
        Data_STDP.Coupled.PC_noise = Data.Coupled.PC_noise(:,1:coldata/2);
        Data_STDP.Coupled.PC_firingrate = Data.Coupled.PC_firingrate(:,1:coldata/2);
        Data_STDP.Coupled.IO_Vs = Data.Coupled.IO_Vs(:,1:coldata/2);
        colIO = size(Data.Coupled.IO_spikes,2);
        for ii = 1:colIO
            col = size(Data.Coupled.IO_spikes{1,ii},2);
            for kk = 1:col
                if Data.Coupled.Noise_t(end)/2000 <Data.Coupled.IO_spikes{1,ii}(kk)
                    break
                end
                Data_STDP.Coupled.IO_spikes{1,ii}(kk) = Data.Coupled.IO_spikes{1,ii}(kk);
            end
        end        
        Data_STDP.Coupled.IO_firingrate = Data.Coupled.IO_firingrate(:,1:coldata/2);
        Data_STDP.Coupled.DCN_v = Data.Coupled.DCN_v(:,1:coldata/2);
        colDCN = size(Data.Coupled.DCN_spikes,2);
        for ii = 1:colDCN
            col = size(Data.Coupled.DCN_spikes{1,ii},2);
            for kk = 1:col
                if Data.Coupled.Noise_t(end)/2000 <Data.Coupled.DCN_spikes{1,ii}(kk)
                    break
                end
                Data_STDP.Coupled.DCN_spikes{1,ii}(kk) = Data.Coupled.DCN_spikes{1,ii}(kk);
            end
        end    
        Data_STDP.Coupled.DCN_firingrate = Data.Coupled.DCN_firingrate(:,1:coldata/2);        
end

if isfield(Params, 'Uncoupled' )    
    load(Params.Uncoupled.mat);
    coldata = size(Data.Uncoupled.Noise_t,2);
    Data_STDP.Uncoupled = struct;
    Data_STDP.Uncoupled.Noise_t = Data.Uncoupled.Noise_t(:,1:coldata/2);
    Data_STDP.Uncoupled.Noise_I = Data.Uncoupled.Noise_I(:,1:coldata/2);
    Data_STDP.Uncoupled.PC_v = Data.Uncoupled.PC_v(:,1:coldata/2);
    colPC = size(Data.Uncoupled.PC_spikes,2);
    for ii = 1:colPC
        col = size(Data.Uncoupled.PC_spikes{1,ii},2);
        for kk = 1:col
            if Data.Uncoupled.Noise_t(end)/2000 <Data.Uncoupled.PC_spikes{1,ii}(kk)
                break
            end
            Data_STDP.Uncoupled.PC_spikes{1,ii}(kk) = Data.Uncoupled.PC_spikes{1,ii}(kk);
        end
    end
    Data_STDP.Uncoupled.PC_noise = Data.Uncoupled.PC_noise(:,1:coldata/2);
    Data_STDP.Uncoupled.PC_firingrate = Data.Uncoupled.PC_firingrate(:,1:coldata/2);
    Data_STDP.Uncoupled.IO_Vs = Data.Uncoupled.IO_Vs(:,1:coldata/2);
    colIO = size(Data.Uncoupled.IO_spikes,2);
    for ii = 1:colIO
        col = size(Data.Uncoupled.IO_spikes{1,ii},2);
        for kk = 1:col
            if Data.Uncoupled.Noise_t(end)/2000 <Data.Uncoupled.IO_spikes{1,ii}(kk)
                break
            end
            Data_STDP.Uncoupled.IO_spikes{1,ii}(kk) = Data.Uncoupled.IO_spikes{1,ii}(kk);
        end
    end        
    Data_STDP.Uncoupled.IO_firingrate = Data.Uncoupled.IO_firingrate(:,1:coldata/2);
    Data_STDP.Uncoupled.DCN_v = Data.Uncoupled.DCN_v(:,1:coldata/2);
    colDCN = size(Data.Uncoupled.DCN_spikes,2);
    for ii = 1:colDCN
        col = size(Data.Uncoupled.DCN_spikes{1,ii},2);
        for kk = 1:col
            if Data.Uncoupled.Noise_t(end)/2000 <Data.Uncoupled.DCN_spikes{1,ii}(kk)
                break
            end
            Data_STDP.Uncoupled.DCN_spikes{1,ii}(kk) = Data.Uncoupled.DCN_spikes{1,ii}(kk);
        end
    end    
    Data_STDP.Uncoupled.DCN_firingrate = Data.Uncoupled.DCN_firingrate(:,1:coldata/2);        
end

end
