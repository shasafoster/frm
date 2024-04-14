function SpotPrice = ClewlowStrickland_1FactorMC_Daily(nSims, Seed, FC, nDays, cashVol, meanRe, Scheme)
                                                       
%**************************************************************************
%
% This module solves numerical the Clewlow and Strickland one factor spot
% model using several different discretizations all based having Delta t
% equal to unity.
%
% Inputs:
%        nSims - number of simulations
%        Seed - seed for random number generator
%        FC - Contains forward curve array at daily detail
%        nDays - number of days
%        cashVol - daily cash volatility (A)
%        meanRe - daily mean reversion of TSOV (c)
%        Schemes - 1 = Euler log transformation 
%                  2 = Euler scheme 
%                  3 = Semi implicit Euler log transformation
%                  4 = Weak predictor/Corrector on log transformation
%                  5 = Iterative scheme based on the Ito integal
%
% Outputs:
%        SpotPrice - Daily spot prices (S)
%
% Ahmos Sansom - February 2010
%**************************************************************************


% initialise random number
randn('seed',Seed);

% allocate spot price
SpotPrice = zeros(nDays,nSims);

switch Scheme
    case 1 % change in log S - Euler discretised form of:
           % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1 day ahead.
        SpotPrice(1,i) = FC(1);

        for j=2:nDays

            % set random step values
            epsilon = randn;

            % set time, start at t=1
            time = j-1;

            expTerm =  exp( cashVol*(-0.25*cashVol * (1.0 + exp(-2.0*meanRe*time)) + epsilon) );
            SpotPrice(j,i) = (SpotPrice(j-1,i)^(1.0 - meanRe)) * FC(j) * (FC(j-1)^(meanRe-1.0)) * expTerm;
        end 
    end
    
    case 2 % change in S - Euler discretised form of:
           % dS/S = [ d ln F / dt + c ( ln F - ln S ) + 0.25 c [1 - exp(-2ct)] dt + A dz 

    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1 day ahead.
        SpotPrice(1,i) = FC(1);

        for j=2:nDays

            % set random step values
            epsilon = randn;

            % set time, start at t=1
            time = j-1;

            Term =  log(FC(j)/FC(j-1)) + meanRe * log(FC(j-1)/SpotPrice(j-1,i)) + 0.25 * cashVol*cashVol * (1.0 - exp(-2.0*meanRe*time));
            SpotPrice(j,i) = SpotPrice(j-1,i) * ( 1.0 + ( Term + cashVol*epsilon));
        end 
    end

   case 3  % change in log S - Semi implicit Euler discretised form of:
           % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1 day ahead.
        SpotPrice(1,i) = FC(1);

        for j=2:nDays

            % set random step values
            epsilon = randn;

            % set time, start at t=1
            time = j-1;

            expTerm =  exp( cashVol*(-0.25*cashVol * (1.0 + exp(-2.0*meanRe*time)) + epsilon) );
            SpotPrice(j,i) = (SpotPrice(j-1,i) * (FC(j)^(1+meanRe)) * (FC(j-1)^(-1.0)) * expTerm)^(1.0/(1.0+meanRe));
        end 
    end
      
    case 4  % change in log S - weak predictor/corrector discretised form of:
            % d ln S = [d ln F / dt + c ( ln F - ln S ) - 0.25 c [1 + exp (-2ct)] dt + A dz    
        
    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1 day ahead.
        SpotPrice(1,i) = FC(1);

        for j=2:nDays

            % set random step values
            epsilon = randn;

            % set time, start at t=1
            time = j-1;

            % Predictor
            expTerm =  exp( cashVol*(-0.25*cashVol * (1.0 + exp(-2.0*meanRe*time)) + epsilon) );
            Predictor = (SpotPrice(j-1,i)^(1.0 - meanRe)) * FC(j) * (FC(j-1)^(meanRe-1.0)) * expTerm;
            
            % Corrector
            expTerm =  exp( cashVol*(-0.25*cashVol * (1.0 + exp(-2.0*meanRe*time)) + epsilon) );
            SpotPrice(j,i) = SpotPrice(j-1,i)*Predictor^(-meanRe) * FC(j) * (FC(j-1)^(meanRe-1.0)) * expTerm;
            
        end 
    end
    
    case 5 % iterative strategy based on stochastic integral (It)
           % S =  F * exp (It - Jt) where Jt is deterministic and It is
           % stochastic: Jt = A^2(1-exp(-2ct))/4c
           %             It = int(A exp(-ct)dW

    % loop round sims
    for i=1:nSims

        % initialise spot price as FC at time t = 1 day ahead.
        SpotPrice(1,i) = FC(1);
        It = 0;

        for j=2:nDays

            % set random step values
            epsilon = randn;

            % set time, start at t=1
            time = j-1;
            
            % Deterministic integral
            Jt = 0.25 * cashVol*cashVol*(1 - exp(-2.0*meanRe*time))/meanRe;
            
            %iterative scheme for It
            sigmaAv = sqrt(0.5 * cashVol*cashVol*(1 - exp(-2.0*meanRe))/meanRe);
            It = exp(-meanRe) * It + epsilon * sigmaAv;
            
            expTerm = exp(It -  Jt);

            SpotPrice(j,i) = FC(j) * expTerm;
        end 
    end    
    
end
           