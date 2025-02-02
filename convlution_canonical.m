function out = convlution_canonical(s,t)
%CONVLUTION Summary of this function goes here
%   Detailed explanation goes here
peakTime    = 4;
uShootTime  = 16;
peakDisp    = 1;
uShootDisp  = 1;
ratio       = 1/6;
duration    = 32;


% params
a1 = peakTime;
a2 = uShootTime;
b1 = peakDisp;
b2 = uShootDisp;
c  = ratio;

% sampling freq
Fs = 1/(t(2)-t(1));

% time vector
t = (0:1/Fs:duration)';

% impulse response
h = getImpulseResponse( a1, b1, a2, b2, c, t );

%             % stupid filtering function
%             f = @(s) bsxfun( @minus, s, s(1,:) );
%             f = @(h, s) filter(h, 1, f(s));
%             f = @(h, s) bsxfun( @plus, f(h,s), s(1,:) );

% convert stim vectors
out = filter(h, 1, s);
end

function h = getImpulseResponse( a1, b1, a2, b2, c, t )
h = b1^a1*t.^(a1-1).*exp(-b1*t)/gamma(a1) - c*b2^a2*t.^(a2-1).*exp(-b2*t)/gamma(a2);
h = h / sum(h);
end

