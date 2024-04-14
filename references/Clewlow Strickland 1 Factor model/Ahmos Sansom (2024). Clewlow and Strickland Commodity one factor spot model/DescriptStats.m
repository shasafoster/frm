function ResultStats = DescriptStats(InData)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         % 
%  Creates an array of simple stats                                       %
%                                                                         %
%  Ahmos Sansom - October 2007                                            %
%                                                                         %
%  Inputs:  InData: data vector                                           %
%                                                                         %
%  Outputs: ResultStats: (min, max, mean, std, skew, kurtosis)            %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get dimensions of array
[rows cols] = size(InData);

if cols > 1 
   error('Error in input - can only process vectors!') 
end 

% Declare array for results
ResultStats = zeros(1,8);

ResultStats(1,1) = min(InData);
ResultStats(1,2) = max(InData);
ResultStats(1,3) = mean(InData);
ResultStats(1,4) = std(InData);
ResultStats(1,5) = skewness(InData);
ResultStats(1,6) = kurtosis(InData);
ResultStats(1,7) = sum(InData);

