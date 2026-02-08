function bpmMask = bipolarMontage(header)
%BIPOLARMONTAGE_LB18  Returns an 18×2 montage mask for LB-18.1 (double banana).
%
% This function implements the classic 18-channel longitudinal bipolar montage:
%   1)  Fp1–F7
%   2)  F7–T7
%   3)  T7–P7
%   4)  P7–O1
%   5)  Fp1–F3
%   6)  F3–C3
%   7)  C3–P3
%   8)  P3–O1
%   9)  Fp2–F8
%   10) F8–T8
%   11) T8–P8
%   12) P8–O2
%   13) Fp2–F4
%   14) F4–C4
%   15) C4–P4
%   16) P4–O2
%   17) Fz–Cz
%   18) Cz–Pz
%
% USAGE:
%   bpmMask = bipolarMontage_LB18(header);
%   d1 = EEGData(bpmMask(:,1),:) - EEGData(bpmMask(:,2),:);
%
% INPUT:
%   header - struct from edfread (header.label is a cell array of channel names)
%
% OUTPUT:
%   bpmMask - An 18x2 array of integer indices. Each row has [leftChanIndex, rightChanIndex]
%             so you can do left minus right for your bipolar channels.

% Define the expected channel pairs:
lbPairs = {
    'EEGFp1Ref','EEGF7Ref';  % (1)  Fp1-F7
    'EEGF7Ref','EEGT7Ref';   % (2)  F7-T7
    'EEGT7Ref','EEGP7Ref';   % (3)  T7-P7
    'EEGP7Ref','EEGO1Ref';   % (4)  P7-O1

    'EEGFp1Ref','EEGF3Ref';  % (5)  Fp1-F3
    'EEGF3Ref','EEGC3Ref';   % (6)  F3-C3
    'EEGC3Ref','EEGP3Ref';   % (7)  C3-P3
    'EEGP3Ref','EEGO1Ref';   % (8)  P3-O1

    'EEGFp2Ref','EEGF8Ref';  % (9)  Fp2-F8
    'EEGF8Ref','EEGT8Ref';   % (10) F8-T8
    'EEGT8Ref','EEGP8Ref';   % (11) T8-P8
    'EEGP8Ref','EEGO2Ref';   % (12) P8-O2

    'EEGFp2Ref','EEGF4Ref';  % (13) Fp2-F4
    'EEGF4Ref','EEGC4Ref';   % (14) F4-C4
    'EEGC4Ref','EEGP4Ref';   % (15) C4-P4
    'EEGP4Ref','EEGO2Ref';   % (16) P4-O2

    'EEGFzRef','EEGCzRef';   % (17) Fz-Cz
    'EEGCzRef','EEGPzRef';   % (18) Cz-Pz
};

% Pre-allocate the 18x2 mask
bpmMask = nan(18,2);

% For each row/pair, find the matching indices in header.label
for i = 1:size(lbPairs,1)
    leftChan  = lbPairs{i,1};
    rightChan = lbPairs{i,2};

    iLeft  = find(strcmp(header.label, leftChan));
    iRight = find(strcmp(header.label, rightChan));

    if isempty(iLeft) || isempty(iRight)
        warning('Channel(s) not found: %s or %s', leftChan, rightChan);
    else
        bpmMask(i,1) = iLeft;
        bpmMask(i,2) = iRight;
    end
end
end

