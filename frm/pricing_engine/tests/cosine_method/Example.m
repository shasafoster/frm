clc
clear

% Examples are trivial and obvious. 
% The code in the COS_method function can be used in much complex
% settings, and adapted to fit several problems.
% Afterall, the main part of the function are  the rows w=, Fk=,C=,f= ...
% dealing with the actual COS method implementation: the rest of the code
% can be easily tuned to your speficif input function and estimation problem.

%% PDF and CDF of a standard normal distirbution

mu = 0;
sig = 1;

% Characteristic function of a standard normal
cf = @(s) exp(1i*s*mu-1/2*sig^2*s.^2);

[Fp,F,f,I,pts] = COS_method(1.644854,cf,100,0.0001,-10,10);

fprintf('CDF(p): %.4f \n',Fp)
% Indeed CDF(1.644854) is 0.95

fprintf('Integral: %.4f \n',I)
% [a,b] are appropriate, the integral over [a,b] of the PDF is one


plot_xp = -10:0.1:10;
subplot(1,2,1)
plot(pts,f,'LineWidth',1.2)
grid
hold on
plot(plot_xp,normpdf(plot_xp,mu,sig),'--r')
hold off
legend({'COS method','Normal pdf'})
subplot(1,2,2)
plot(pts,F)
grid
hold on
plot(plot_xp,normcdf(plot_xp,mu,sig),'--r')
hold off
legend({'COS method','Normal pdf'})

%% Median of a Normal(8,2)

mu = 8;
sig = 2;

% Characteristic function of a standard normal
cf = @(s) exp(1i*s*mu-1/2*sig^2*s.^2);

Fp = COS_method(8,cf,100,0.0001,-0,16);

fprintf('CDF(p): %.4f \n',Fp)
% Indeed CDF(8) is 0.50, 8 is the median



%% Compute expected value using MGF
% estimate the expectation of Y = 3+5*X, where X ~ Gamma(a,b).
% The MGF is given by exp(3*t)*M(5*t), where M is the MGF of X.
% Get the CF from the MGF, CF(t) = MGF(1i*t).
% Compute the expectation by sum(x*f*xd)


a = 5;
b = 8;

mgf = @(t) exp(3.*t).*(1-5*b*t).^-a;
cf  = @(t) mgf(1i*t);

[Fp,F,f,I,pts] = COS_method(0,cf,200,0.05,0,1200);

m = sum(f.*pts*0.05);

fprintf('Mean: %.2f \n',m)
% Theoretical expectation: 3+5*(a*b) = 203

%%

