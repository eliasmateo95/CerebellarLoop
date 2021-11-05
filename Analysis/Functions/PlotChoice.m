function [Params] = PlotChoice(Params,TYPE,NOISE,PCNOISE,ALLNOISE,IODCNPC,PC,DCN,IO,CSTRIGNOISE,CSTRIGSS,CSTRIGDCN,DCNTRIGNOISE,FIRING,CORRFIRING,RASTER,NOISECOV,PCCOV,WATERFALLNOISE,WATERFALLIOVS)
if TYPE == "Coupled"
    Params.Plot.Coupled = struct;
    Params.Plot.Coupled.Noise = NOISE;
    Params.Plot.Coupled.PCNoise = PCNOISE;
    Params.Plot.Coupled.AllNoise = ALLNOISE;
    Params.Plot.Coupled.IODCNPC = IODCNPC;
    Params.Plot.Coupled.PC = PC;
    Params.Plot.Coupled.DCN = DCN;
    Params.Plot.Coupled.IO = IO;
    Params.Plot.Coupled.CSTrigNoise = CSTRIGNOISE;
    Params.Plot.Coupled.CSTrigSS = CSTRIGSS;
    Params.Plot.Coupled.CSTrigDCN = CSTRIGDCN;
    Params.Plot.Coupled.DCNTrigNoise = DCNTRIGNOISE;
    Params.Plot.Coupled.firing = FIRING;
    Params.Plot.Coupled.corrfiring = CORRFIRING;
    Params.Plot.Coupled.raster = RASTER;
    Params.Plot.Coupled.Noisecov = NOISECOV;
    Params.Plot.Coupled.PCcov = PCCOV;
    Params.Plot.Coupled.waterfallNoise = WATERFALLNOISE;
    Params.Plot.Coupled.waterfallIOVs = WATERFALLIOVS;
elseif TYPE == "Uncoupled"
    Params.Plot.Uncoupled = struct; 
    Params.Plot.Uncoupled.Noise = NOISE;
    Params.Plot.Uncoupled.PCNoise = PCNOISE;
    Params.Plot.Uncoupled.AllNoise = ALLNOISE;
    Params.Plot.Uncoupled.IODCNPC = IODCNPC;
    Params.Plot.Uncoupled.PC = PC;
    Params.Plot.Uncoupled.DCN = DCN;
    Params.Plot.Uncoupled.IO = IO;
    Params.Plot.Uncoupled.CSTrigNoise = CSTRIGNOISE;
    Params.Plot.Uncoupled.CSTrigSS = CSTRIGSS;
    Params.Plot.Uncoupled.CSTrigDCN = CSTRIGDCN;
    Params.Plot.Uncoupled.DCNTrigNoise = DCNTRIGNOISE;
    Params.Plot.Uncoupled.firing = FIRING;
    Params.Plot.Uncoupled.corrfiring = CORRFIRING;
    Params.Plot.Uncoupled.raster = RASTER;
    Params.Plot.Uncoupled.Noisecov = NOISECOV;
    Params.Plot.Uncoupled.PCcov = PCCOV;
    Params.Plot.Uncoupled.waterfallNoise = WATERFALLNOISE;
    Params.Plot.Uncoupled.waterfallIOVs = WATERFALLIOVS;
end
end