function[Fp,F,f,I,pts] = COS_method(p,cf,N,dt,a,b)

% PROBLEM STATEMENT: The charactheristic function (or moment generating function, or
% Laplace transform, from which to retrieve it) of a certain random variable X is known. 
% The objective is approximating f and F, respectively PDF and CDF of X.

% INPUTS:
% cf:   charachteristic function provided as a function handle where the only variable is the point at which to evaluate it. 
%       (See the examples). 
% p:    point at which to evaluate the CDF (F) of f - must be in [a,b].
% N:    Number of sums
% dt:   Step size for approximating the integral
% a:    Lower integration bound
% b:    Upper integration bound
%
% OUTPUTS:
% Fp:   CDF evaluated at p
% F:    CDF evaluated on pts
% f:    PDF evaluated on pts
% I:    Integral of the PDF over [a,b]: if ~=1, increase the width of [a,b]
% pts:  points over which the CDF and PDF are sampled: a:dt:pts (and point p)

pts     = sort(unique([p,a:dt:b]));
dt_pts  = [diff(pts),dt];
Dt      = range(pts);
k       = 0:N;

w   = [0.5,ones(1,N)];
Fk  = 2/Dt*real(cf(k*pi/Dt).*exp(-1i*k*pi*a/Dt));
C   = @(x) cos(k*pi*(x-a)/Dt);
f   = arrayfun(@(x) sum(w.*(Fk.*C(x))),pts);

I   = sum(f.*dt_pts);
F   = cumsum(f.*dt_pts);
Fp  = F(pts == p);

end