%function Result = MCControlDailySpotPrice(nSims,Scheme,Strips,Seed,CashVol,MeanRev,nDays)
%
% Can be made as a function to batch results if required
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Control code for Clewlow Strickland 1 factor daily spot model
%
% This code simulates commodity spot prices using the Clewlow and
% Strickland one factor daily spot model.
%
% The paper detailing the equations is available online in ref 1 below.  
%
% The example requires a commodity forward curve and assumes a one factor
% volatility model of the form sigma = A exp(-c(T-t)), where A is the cash
% volatility, c is the mean reversion rate and T is the maturity.
%
% The code highlights several different finite difference schemes to solve
% the spot equation. Note that other methods can be applied.
%
% Ahmos Sansom - February 2010
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%% Initialise inputs

% start stop watch
tic;

% Number of simulated price paths
nSims = 10000;

% Seed for the random number generator - need to be careful using a seed.
Seed = 1234;

% Load in FC daily data from FCDat.mat file
load FCData;

% number of days to simulate daily spot price - must be less or equal to FC
% input
nDays = 750;

% Volatility parameters:
% Cash volatility (daily)
CashVol = 0.04;
% Mean reversion (daily)
MeanRev = 0.06;

% Numerical finite difference schemes
%  Schemes - 1 = Euler log transformation 
%            2 = Euler scheme 
%            3 = Semi implicit Euler log transformation
%            4 = Weak predictor/Corrector on log transformation
Scheme = 1;

% Number of strips per day, need to interpolated FC data in order to match
% the requred time step in order to reduce discretization bias. Delta t =
% 1/Strips. Increase for accuarcy.
Strips = 5;

% Get simulated daily spot paths
SpotPrice = ClewlowStrickland_1FactorMC_DailyInterp(nSims, Seed, FC, nDays, CashVol, MeanRev, Strips, Scheme);

% Can use the follwing routine without the interplation to validate results
% at daily intervals (i.e. Strips = 1).
%SpotPrice = ClewlowStrickland_1FactorMC_Daily(nSims, Seed, FC, nDays, CashVol, MeanRev, Scheme);

% Get some descriptive stats
SpotPriceStats = SimDescriptStats(SpotPrice,FC);

% Output time taken
fprintf('Simulation time: %6.1f minutes\n',toc/60.0);
%% Validation
% The spot price paths can be validated using european call and put option
% valuations. Validation assumes an Asian option based on the last 729
% days.

% Set strike price and maturity
Strike = FC(1);
Maturity = 730;

% Back test for one day up to maturity.

% Initialise
CallValueAnalytical=zeros(Maturity,1);
PutValueAnalytical =zeros(Maturity,1);

% Analytical formula for a standard European call and put option from Black
% and Scholes - see equation 3.6 in ref [1].
for i=1:Maturity
    time = 0;

    w = 0.5*CashVol*CashVol*(1.0-exp(-2.0*MeanRev*(i - time)))/MeanRev;
    h = (log(FC(i)/Strike)+0.5*w)/sqrt(w);
    CallValueAnalytical(i) = FC(i)*normcdf(h,0,1)-Strike*normcdf(h-sqrt(w),0,1);
    PutValueAnalytical(i) = Strike*normcdf(-h+sqrt(w),0,1)-FC(i)*normcdf(-h,0,1);
end

% Compare with Monte Carlo
CallValueMonte=zeros(Maturity,1);
PutValueMonte =zeros(Maturity,1);

for j=1:Maturity
    for i=1:nSims
        if SpotPrice(j,i) > Strike
           CallValueMonte(j) = CallValueMonte(j) + SpotPrice(j,i) - Strike;
        else
           PutValueMonte(j) = PutValueMonte(j) +  Strike - SpotPrice(j,i);
        end        
    end
end

CallValueMonte = CallValueMonte./nSims;
PutValueMonte = PutValueMonte./nSims;

% Get some stats
CallAnalytical = mean(CallValueAnalytical(2:730));
CallMonte = mean(CallValueMonte(2:730));
PutAnalytical = mean(PutValueAnalytical(2:730));
PutMonte = mean(PutValueMonte(2:730));

fprintf('\nCall Analytical valuation = %6.4f compared with Monte Carlo = %6.4f\n', CallAnalytical, CallMonte);
fprintf('Put Analytical valuation = %6.4f compared with Monte Carlo = %6.4f\n\n', PutAnalytical, PutMonte);

Result(1,1) = CallMonte;
Result(1,2) = PutMonte;

%% References

% Reference 1 details the derivation of the one factor model that is
% detailed further in Clewlow and Strickland's book referenced in 2. This 
% books is available in pdf from www.lacimagroup.com and the website has 
% available many papers to freely download discussing commodities.

% 1. http://ideas.repec.org/p/uts/rpaper/10.html 
% 2. "Energy Derivatives: Pricing and Risk Management," Clewlow and
% Strickland, Lacima Group, 2000.
