function MAPEArray = MAPEcalc(SimArray, RealValues)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         % 
%  Creates an array of MAPE values from sims                              %
%                                                                         %
%  Ahmos Sansom - July 2008                                               %
%                                                                         %
%  Inputs:  SimArray: Array with simulations                              %
%           RealValues: historical values                                 %
%                                                                         %
%  Outputs: MAPEArray: array of MAPEs for each discrete simulation (n)    %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get sim size
n = size(SimArray,1);
nSims = size(SimArray,2);

% assign memory
MAPEArray = zeros(nSims,1);

% Calculate the MAPE for each simulation at each n.
for i = 1 : nSims
    for j = 1 : n
        if RealValues(j) ~= 0
   
           temp = abs( ( RealValues(j) - SimArray(j,i) ) / RealValues(j)) ;
           MAPEArray(i) = MAPEArray(i) + temp;
        end
    end
    MAPEArray(i) = MAPEArray(i) / n;
end
