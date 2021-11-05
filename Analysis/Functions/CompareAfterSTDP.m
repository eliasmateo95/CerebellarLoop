function CompareAfterSTDP(Data,Data_AfterSTDP,Params)
show = Params.show;
save = Params.save;

[r,lags] = xcorr(Data.Coupled.PC_firingrate,Data_AfterSTDP.Coupled.PC_firingrate,'unbiased');
h = figure('Renderer', 'painters', 'visible', show, 'Position', [10 50 1500 710]);
subplot(2,1,1)
plot(lags/(10^6),r/(10^3))
title('PC_{No STDP} vs. PC_{STDP} Rates Cross-correlated Coupled'+Name)
subplot(2,1,2)
[r1,lags1] = xcorr(Data.Uncoupled.PC_firingrate,Data_AfterSTDP.Uncoupled.PC_firingrate,'unbiased');
plot(lags1/(10^6),r1/(10^3))
title('PC_{No STDP} vs. PC_{STDP} Rates Cross-correlated Uncoupled '+Name)
if save == "True"
    saveas(h, fullfile(fname,'PC_Before_After_STDP_Rates_Xcorr'), 'jpeg');
end 

end