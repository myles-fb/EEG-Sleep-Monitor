Below is the Matlab code from the pre-existing pipeline for the lasso calculation. Please update the codebase to reflect the approach used in this method -- q-values should be chosen based on max, not mean, and there should be a q-value for each window, not just each bucket.

function [qm] = lassoHIE(env,T1,fname,vars)
%DOMINANT FREQUENCY TRACKING:
% f_strength variable: For each window and channel, after obtaining the coefficients b, 
% the code loops over each candidate frequency (defined in a_freqs) 
% and for each regularization solution (lambda) computes a “strength” 
% by combining the sine and cosine coefficients in a polar coordinate–like calculation. 
% The resulting magnitude (the first element of polar_vec) is stored in f_strength. 
% This value reflects how strongly a particular sinusoid (i.e., candidate frequency) is present 
% in the model for that window.
%

% function [qm,qv] = lassoHIE(env,T1,fname,vars)
wintime = vars(1); % 300 seconds
Fs = vars(2);
winsize = round(vars(3)); % 300 seconds * (1/(T1(1) - T1(2)) i.e. spectral envelope sampling freq
winjump = vars(4); %step size (winsize/4)
numchan = size(env,2); %18
%dictionary of sinusoids
a_freqs = (2/wintime):(.5/wintime):(Fs/4);
%   "Our dictionary of sinusoids spanned... 
%   from fmin = 2*fs/Nm to fmax = fs/4... harmonic components every
%   fmin/4 Hz" (pg 5). This does not match what I'm seeing here.
%   1/wintime = sampling frequency (maximum freq resolution I can detect).
%   Range of freqs for sinusoidal dictionary is 2*fs
    t0 = T1(1:round(winsize)); %time_steps for 1 window
    alph = 2;
    A = nan(length(t0),alph*length(a_freqs)); %row per time_step, columns for number of candidate frequencies defined in a_freq * 2 (sin, cos)
    for k=1:length(a_freqs) %outer for-loop iterating through all candidate frequencies in dictionary
        for j=1:alph
            omega = a_freqs(k)*2*pi; %converting to radians
            % A(:, etc) is sinusoid down an entire column that represents a time window
            % (for channel, freq band)
            % 2*pi*j/alpha yield pi and 2*pi depending on j. This yields
            % sin(wt) and -sin(wt) which are linearly dependent.
            % 3/23/2025: Adjusted inner argument so that phase shifts are
            % 2pi and pi/2. (Added parentheses).
            % UPDATE: Changed it back because there might be a benefit for
            % only having sin() -sin() as basis functions; phase isn't
            % considered when computing entropy.
            %sin(wt) and -sin(wt) are stored in adjacent columns
            A(:,(k-1)*alph+j) = reshape(sin(t0*omega+2*pi/alph*j),length(t0),1);
        end
    end
thetas = reshape((2*pi/alph:2*pi/alph:2*pi),alph,1); %capture phase shifts applied in dictionary. originally: pi, 2pi.
num_funs = floor((length(env)-winsize)/winjump); %number of windows
f_strength = cell(num_funs+1,numchan); % frequencies x lambdas x times x channels

b = cell(numchan,num_funs+1);
bs = cell(numchan,num_funs+1);
J = cell(num_funs+1,numchan);
q_entropy=J;
q = J;

s = movmean(env,10,1);% filter whole signal
qm = nan(num_funs,numchan); 
% end

% parpool('local')
for ch=1:numchan
% num_funs = floor((length(s)-window_sz)/winjump);
f_hold=[];
parfor t=1:num_funs
w0 = round((1:winsize) +(t-1)*winjump);
s0 = 2*(s(w0,ch)-min(s(w0,ch)))./range(s(w0,ch))-1;
s0 = s0-mean(s0);

[bb,ss] = lasso(A,s0);
b{ch,t} = bb;
bs{ch,t}=ss;
J{t,ch} = corr(A*bb,s0);
f_hold = nan(length(a_freqs),size(bb,2),num_funs+1);
% convert strengths + take entropy of strengths
for lambda=1:size(bb,2)
    for fr=1:length(a_freqs) %across frequencies fitted
    v = (1:alph)+alph*(fr-1); %identify frequencies
    fitted_weights = bb(v,lambda);
    grid_coords = [fitted_weights.*cos(thetas),fitted_weights.*sin(thetas)];
    xy_vec = sum(grid_coords);
    polar_vec = [sqrt(sum(xy_vec.^2)), atan(xy_vec(2)/xy_vec(1))];
    f_hold(fr,lambda,t) = polar_vec(1);
    end % all freqs
end % all lambdas
q_entropy{t,ch} = take_entropy_lasso(f_hold(:,:,t));
f_strength{t,ch} = f_hold(:,:,t);
q{t,ch} = J{t,ch}.*(1-q_entropy{t,ch});

qm(t,ch)=max(q{t,ch});
[~, qi] = max(q{t,ch});
lambda_cube(t,ch)=ss.Lambda(qi);
idx_cube(t,ch)=qi;
wts_cube(:,ch,t) = bb(:,qi);
bb=[];
ss=[];
qi=[];
f_hold=[];
end %t=1:end-1;
t=num_funs+1;
w0 = (1:winsize) + length(s)-winsize;
s0 = 2*(s(w0,ch)-min(s(w0,ch)))./range(s(w0,ch))-1;
s0 = s0-mean(s0);
[bb,ss] = lasso(A,s0);
b{ch,t} = bb;
bs{ch,t}=ss;
J{t,ch} = corr(A*bb,s0);
% convert strengths + take entropy of strengths
    for lambda=1:size(bb,2)
    for fr=1:length(a_freqs) %across frequencies fitted
    v = (1:alph)+alph*(fr-1); %identify frequencies
    fitted_weights = bb(v,lambda); %this yields structure frequency x weights
    grid_coords = [fitted_weights.*cos(thetas),fitted_weights.*sin(thetas)];
    xy_vec = sum(grid_coords);
    polar_vec = [sqrt(sum(xy_vec.^2)), atan(xy_vec(2)/xy_vec(1))];
    f_hold(fr,lambda,t) = polar_vec(1);
    end %all freqs
    end % all lambdas
    q_entropy{t,ch} = take_entropy_lasso(f_hold(:,:,t));
    f_strength{t,ch} = f_hold(:,:,t);

q{t,ch} = J{t,ch}.*(1-q_entropy{t,ch});

qm(t,ch)=max(q{t,ch}); %
bb=[];
ss=[];
end % all channels
% clear env

save(fname,'b','bs','J','q','qm','q_entropy','f_strength')
% save([savenamebase,'_lasso_',num2str(freq(1)),'_',num2str(freq(2))],'b','bs','J','q','q_80','q_entropy','f_strength')
clear('b','bs','J','q','env','w0','s0','s','q_entropy','f_strength')%,'qch','qm')
% disp(horzcat('finished in hours = ',num2str(toc(tstart0)./3600)))
end

function [y] = take_entropy_lasso(x)
%TAKE_ENTROPY Summary of this function goes here
%   Detailed explanation goes here
x = abs(x);
[a, b, c] = size(x);
prob_dist = x./sum(x);
y = reshape(nansum(-1.*prob_dist.*log(prob_dist))./log(a),b,c);
end

