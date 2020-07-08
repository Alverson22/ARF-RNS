function [feature,result,DimFeature,NumTrainingSample] = getTrainingFeatureAndLabel(Mode,RealData,ImagData,TrainingTimeStep,PredictTimeStep)
% This function is to
%   1. transform the received packets to feature vectors for training
%      data collection;
%   2. collect the corresponding labels.

% Determine the feature size
% NumTrainingSample = floor((size(RealData,1) - (TrainingTimeStep+PredictTimeStep-1)) / TrainingDataInterval;
NumTrainingSample = floor((size(RealData,1) - (TrainingTimeStep+PredictTimeStep)) / PredictTimeStep);

% Data Collection
realDiff = RealData(2:end) - RealData(1:end-1); 
imagDiff = ImagData(2:end) - ImagData(1:end-1); 

% The LEO CSI need to be normalized to allow GRU learning model to learn
realDiff_MEAN = mean(realDiff);
realDiff_STD = std(realDiff);
RealPart = (realDiff-realDiff_MEAN) / realDiff_STD;
imagDiff_MEAN = mean(imagDiff);
imagDiff_STD = std(imagDiff);
ImagPart = (imagDiff-imagDiff_MEAN) / imagDiff_STD;


% Generating training sequence only CSI signal
if Mode == 'S'
    % Feature vector
    DimFeature = 2;
    DimFeatureVec = TrainingTimeStep * DimFeature; % real + imag
    DimResultVec = PredictTimeStep * DimFeature;
    feature = zeros(DimFeatureVec,NumTrainingSample);
    result = zeros(DimResultVec,NumTrainingSample);
    
    currentidx = 1;

    for n = 1:NumTrainingSample
        feature(1:2:end,n) = RealPart(currentidx:currentidx+TrainingTimeStep-1);
        feature(2:2:end,n) = ImagPart(currentidx:currentidx+TrainingTimeStep-1);
        result(1:2:end,n) = RealPart(currentidx+TrainingTimeStep:currentidx+TrainingTimeStep+PredictTimeStep-1);
        result(2:2:end,n) = ImagPart(currentidx+TrainingTimeStep:currentidx+TrainingTimeStep+PredictTimeStep-1);
        currentidx = currentidx + PredictTimeStep;
    end

save('Normalized.mat','realDiff_MEAN','realDiff_STD','imagDiff_MEAN','imagDiff_STD');

end

