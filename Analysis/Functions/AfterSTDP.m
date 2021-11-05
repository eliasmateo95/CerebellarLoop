function [Data_AfterSTDP] = AfterSTDP(Data,Params)
Data_AfterSTDP = struct;
if isfield(Params, 'Coupled' )
        load(Params.Coupled.mat);
        coldata = size(Data.Coupled.Noise_t,2);
        Data_AfterSTDP.Coupled = struct;
        Data_AfterSTDP.Coupled.Noise_t = Data.Coupled.Noise_t(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Coupled.Noise_I = Data.Coupled.Noise_I(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Coupled.PC_v = Data.Coupled.PC_v(:,(coldata/2 + 1):coldata);
        colPC = size(Data.Coupled.PC_spikes,2);
        for ii = 1:colPC
            col = size(Data.Coupled.PC_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Coupled.PC_spikes{1,ii}(kk)>=Data.Coupled.Noise_t(end)/2000  
                    counter = counter + 1;
                    Data_AfterSTDP.Coupled.PC_spikes{1,ii}(counter) = Data.Coupled.PC_spikes{1,ii}(kk);
                end
            end
        end
        Data_AfterSTDP.Coupled.PC_noise = Data.Coupled.PC_noise(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Coupled.PC_firingrate = Data.Coupled.PC_firingrate(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Coupled.IO_Vs = Data.Coupled.IO_Vs(:,(coldata/2 + 1):coldata);
        colIO = size(Data.Coupled.IO_spikes,2);
        for ii = 1:colIO
            col = size(Data.Coupled.IO_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Coupled.IO_spikes{1,ii}(kk)>=Data.Coupled.Noise_t(end)/2000
                    counter = counter + 1;
                    Data_AfterSTDP.Coupled.IO_spikes{1,ii}(counter) = Data.Coupled.IO_spikes{1,ii}(kk);
                end
            end
        end        
        Data_AfterSTDP.Coupled.IO_firingrate = Data.Coupled.IO_firingrate(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Coupled.DCN_v = Data.Coupled.DCN_v(:,(coldata/2 + 1):coldata);
        colDCN = size(Data.Coupled.DCN_spikes,2);
        for ii = 1:colDCN
            col = size(Data.Coupled.DCN_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Coupled.DCN_spikes{1,ii}(kk)>=Data.Coupled.Noise_t(end)/2000
                    counter = counter + 1;
                    Data_AfterSTDP.Coupled.DCN_spikes{1,ii}(counter) = Data.Coupled.DCN_spikes{1,ii}(kk);
                end
            end
        end    
        Data_AfterSTDP.Coupled.DCN_firingrate = Data.Coupled.DCN_firingrate(:,(coldata/2 + 1):coldata);        
end

if isfield(Params, 'Uncoupled' ) 
        load(Params.Uncoupled.mat);
        coldata = size(Data.Uncoupled.Noise_t,2);
        Data_AfterSTDP.Uncoupled = struct;
        Data_AfterSTDP.Uncoupled.Noise_t = Data.Uncoupled.Noise_t(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Uncoupled.Noise_I = Data.Uncoupled.Noise_I(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Uncoupled.PC_v = Data.Uncoupled.PC_v(:,(coldata/2 + 1):coldata);
        colPC = size(Data.Uncoupled.PC_spikes,2);
        for ii = 1:colPC
            col = size(Data.Uncoupled.PC_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Uncoupled.PC_spikes{1,ii}(kk)>=Data.Uncoupled.Noise_t(end)/2000  
                    counter = counter + 1;
                    Data_AfterSTDP.Uncoupled.PC_spikes{1,ii}(counter) = Data.Uncoupled.PC_spikes{1,ii}(kk);
                end
            end
        end
        Data_AfterSTDP.Uncoupled.PC_noise = Data.Uncoupled.PC_noise(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Uncoupled.PC_firingrate = Data.Uncoupled.PC_firingrate(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Uncoupled.IO_Vs = Data.Uncoupled.IO_Vs(:,(coldata/2 + 1):coldata);
        colIO = size(Data.Uncoupled.IO_spikes,2);
        for ii = 1:colIO
            col = size(Data.Uncoupled.IO_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Uncoupled.IO_spikes{1,ii}(kk)>=Data.Uncoupled.Noise_t(end)/2000
                    counter = counter + 1;
                    Data_AfterSTDP.Uncoupled.IO_spikes{1,ii}(counter) = Data.Uncoupled.IO_spikes{1,ii}(kk);
                end
            end
        end        
        Data_AfterSTDP.Uncoupled.IO_firingrate = Data.Uncoupled.IO_firingrate(:,(coldata/2 + 1):coldata);
        Data_AfterSTDP.Uncoupled.DCN_v = Data.Uncoupled.DCN_v(:,(coldata/2 + 1):coldata);
        colDCN = size(Data.Uncoupled.DCN_spikes,2);
        for ii = 1:colDCN
            col = size(Data.Uncoupled.DCN_spikes{1,ii},2);
            counter = 0;
            for kk = 1:col
                if Data.Uncoupled.DCN_spikes{1,ii}(kk)>=Data.Uncoupled.Noise_t(end)/2000
                    counter = counter + 1;
                    Data_AfterSTDP.Uncoupled.DCN_spikes{1,ii}(counter) = Data.Uncoupled.DCN_spikes{1,ii}(kk);
                end
            end
        end    
        Data_AfterSTDP.Uncoupled.DCN_firingrate = Data.Uncoupled.DCN_firingrate(:,(coldata/2 + 1):coldata);        
end

end
