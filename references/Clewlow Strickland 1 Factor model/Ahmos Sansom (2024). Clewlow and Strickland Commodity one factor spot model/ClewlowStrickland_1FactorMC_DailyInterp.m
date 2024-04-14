function SpotPrice = ClewlowStrickland_1FactorMC_DailyInterp(nSims, Seed, FC, nDays, cashVol, meanRe, DailySegs, Scheme)
                                                       
%**************************************************************************
%
% This module solves numerical the Clewlow and Strickland one factor spot
% model using several different discretizations all based having Delta t
% NOT equal to unity.
%
% Inputs:
%        nSims - number of simulations
%        Seed - seed for random number generator
%        FC - Contains forward curve array at daily detail
%        nDays - number of days
%        cashVol - daily cash volatility (A)
%        meanRe - daily mean reversion of TSOV (c)
%        DailySegs - number of segments to split each day
%        Schemes - 1 = Euler log transformation 
%                  2 = Euler scheme 
%                  3 = Semi implicit Euler log transformation
%                  4 = Weak predictor/Corrector on log transformation
%
% Outputs:
%        SpotPrice - Daily spot prices (S)
%
% Ahmos Sansom - February 2010
%**************************************************************************

% Check input data
if rem(DailySegs,1) ~= 0
   error('DailySegs must be an integer') 
end    
if rem(nDays,1) ~= 0
   error('nDays must be an integer') 
end    

% Calculate DT
DT = 1.0/(DailySegs);

% Calculate number of time marches
nSteps = (nDays-1)*DailySegs+1;

% linearly interpolate FC
days = 1:nDays;
days = days';

% Allocate interpolated days
daysInterp = zeros(nSteps,1);
for i=1:nSteps
    daysInterp(i,1) = 1 + (i-1)*DT;
end

FCInterp = interp1(days,FC(1:nDays),daysInterp);

% Create index to obtain daily spot prices 
IndexSpot = zeros(nDays,1);
for i=1:nDays
    IndexSpot(i,1) = 1 +(i-1)*DailySegs;
end

% initialise random number
randn('seed',Seed);

% allocate spot price
SpotPriceTemp = zeros(nSteps,1);
SpotPrice = zeros(nDays,nSims);

switch Scheme
    case 1 % change in log S - Euler discretised form of:
           % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1.
        SpotPriceTemp(1,1) = FCInterp(1);

        for j=2:nSteps

            % set random step values
            epsilon = randn*sqrt(DT);

            % set time, start at t=1
            time = (j-1)*DT;
            
            Term1 = log(FCInterp(j)/FCInterp(j-1))/DT + meanRe * log(FCInterp(j-1)/SpotPriceTemp(j-1,1));
            Term2 = -0.25*cashVol*cashVol*(1.0 + exp(-2.0*meanRe*time));
            expTerm =  exp( (Term1 + Term2)*DT + epsilon*cashVol);
            
            SpotPriceTemp(j,1) = SpotPriceTemp(j-1,1) *  expTerm;
            
        end
        
        % Get daily spot price
        SpotPrice(:,i) = SpotPriceTemp(IndexSpot,1);
        
    end
    
    case 2 % change in S - Euler discretised form of:
           % dS/S = [ d ln F / dt + c ( ln F - ln S ) + 0.25 c [1 - exp(-2ct)] dt + A dz 

    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1.
        SpotPriceTemp(1,1) = FCInterp(1);

        for j=2:nSteps

            % set random step values
            epsilon = randn*sqrt(DT);

            % set time, start at t=1
            time = (j-1)*DT;

            Term1 = log(FCInterp(j)/FCInterp(j-1))/DT + meanRe * log(FCInterp(j-1)/SpotPriceTemp(j-1,1));
            Term2 = 0.25*cashVol*cashVol*(1.0 - exp(-2.0*meanRe*time));            
                        
            SpotPriceTemp(j,1) = SpotPriceTemp(j-1,1) * ( 1.0 + ( (Term1 + Term2)*DT + cashVol*epsilon));
        end
        
        % Get daily spot price
        SpotPrice(:,i) = SpotPriceTemp(IndexSpot,1);
        
    end

   case 3  % change in log S - Semi implicit Euler discretised form of:
           % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1.
        SpotPriceTemp(1,1) = FCInterp(1);

        for j=2:nSteps

            % set random step values
            epsilon = randn*sqrt(DT);

            % set time, start at t=1
            time = (j-1)*DT;

            Term1 = log(FCInterp(j)/FCInterp(j-1))/DT + meanRe * log(FCInterp(j));
            Term2 = -0.25*cashVol*cashVol*(1.0 + exp(-2.0*meanRe*time));
            expTerm =  exp( (Term1 + Term2)*DT + epsilon*cashVol);
            
            SpotPriceTemp(j,1) = (SpotPriceTemp(j-1,1) *  expTerm)^(1.0/(1.0+meanRe*DT));
            
        end
        
        % Get daily spot price
        SpotPrice(:,i) = SpotPriceTemp(IndexSpot,1);
        
    end
      
    case 4  % change in log S - weak predictor/corrector discretised form of:
            % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1.
        SpotPriceTemp(1,1) = FCInterp(1);

        for j=2:nSteps

            % set random step values
            epsilon = randn*sqrt(DT);

            % set time, start at t=1
            time = (j-1)*DT;

            % Predictor
            Term1 = log(FCInterp(j)/FCInterp(j-1))/DT;
            Term2 = meanRe * log(FCInterp(j-1)/SpotPriceTemp(j-1,1));
            Term3 = -0.25*cashVol*cashVol*(1.0 + exp(-2.0*meanRe*time));                                    
            Predictor  = SpotPriceTemp(j-1,1) * exp( (Term1 + Term2 + Term3)*DT + epsilon*cashVol);            
            
            % Corrector
            Term2 = meanRe * log(FCInterp(j-1)/Predictor);                
            SpotPriceTemp(j,1) = SpotPriceTemp(j-1,1) * exp( (Term1 + Term2 + Term3)*DT + epsilon*cashVol);
         
        end
        
        % Get daily spot price
        SpotPrice(:,i) = SpotPriceTemp(IndexSpot,1);
        
    end
    
    
end
           