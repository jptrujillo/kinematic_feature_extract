%%
%%IMPORT DATA, CONVERT TO MATRIX
%%Structure Cm contains one field per joint, with columns for (1)timestamp,
%(2)X-coordinate, (3) Y-coordinate, (4) Z coordinate; each row is a
%separate acquisition
%array TrialdurB contains an array indicating the number of whole body
%acquisitions in each trial (each element corresponds to one trial)
%Takes tab-delimited data

function [Cm, trialdurB,ms] = KFE_import(varargin)
filename = varargin{1};
datatrials = varargin{2};
if length(varargin)>2,
    trialorder = load(varargin{3});
end
clear varargin
%%
%%import data from file
fprintf('\nImporting data from: %s',filename);  

if datatrials.set ==2,
    trialboundsfile = load([datatrials.onoffpath datatrials.onoffname]);%open onsets/offsets file, convert to trialbounds variable
    trialbound(:,1) = trialboundsfile(:,1);
    trialbound(:,3) = trialboundsfile(:,2);
end


alldata = textread(filename,'%s','delimiter','\t');
if datatrials.set==1,
    alldata = KFE_restruct_file(alldata);
end
fprintf('\nConverting to matrix..');  
%%
if datatrials.header ==1,
    alldata = alldata(236:end,1);
end
%convert to Nx4 array, where N is the N is the number of joint acquisitions
if rem(length(alldata)./76,1) == 0,%assumes 1D array, with one timestamp per body acquisition
    S = str2num(alldata{1}); %get first timestamp
    alldataN(1,1) = S;
    j=1;a =1;
    ix = 2;
    for i = 2:length(alldata),
%        if rem(a,77) == 0,
        if a == 76,
           S = str2num(alldata{i});%get new timestamp
           a = 1;
       else
           alldataN(j,ix) = str2num(alldata{i});
           alldataN(j,1) = S;
           a=a+1;
           if ix <4,
               ix = ix+1;
           else
               ix = 2; j = j+1;
           end   
       end       
    end
else
    alldataN = alldata;
end
% clear alldata
   %77 153       
    
% if datatrials.header ==1,
%     if length(alldata(1,:)) <2,
%         if rem((length(alldata)-235)./76,1) ==0,%check format
%             alldataM = (reshape(alldata(236:end,:),[76,(length(alldata)-235)./76]))';%convert to matrix format; necessary for tab-delimited Presentation output
%         else
%     end
% else
%     alldataM = (reshape(alldata(236:end,:),[76,(length(alldata))./76]))';%convert to matrix format; necessary for tab-delimited Presentation output
% end

%check structure



%dynamic struct field reference
jx = cell(1,25);
for i = 1:25,
    jx(i) = {strcat('j',  num2str(i-1))};
end    

if datatrials.set ==1 || datatrials.set==2,
    if datatrials.set ==1,    
        %%
        %%find all start and end marks
        a = 1;
        for i = 1:length(alldataN),
            if strcmp(alldataN{i,1}, datatrials.Sbound),
                %xx(a,1) = alldata{i-1,1};
                xx(a,1) = i; %find the index for each marker
                xx(a,2) = 0;
                a = a+1;
            elseif strcmp(alldataN{i,1}, datatrials.Ebound),
                %xx(a,1) = alldata{i-1,1};
                xx(a,1) = i;
                xx(a,2) = 1;
                a = a+1;
            end
        end

        %%
        %%remove excess marks (caused by high fps of sensor during keydown)
        a = 1;
        for i = 1:(length(xx))-1,
            if xx(i,2) == 0 && xx(i+1,2) == 1, %find occurences of a start followed by an end
                trialbound(a,2) = 0; %only these starts are carried to the new variable
                trialbound(a,4) = 1; %only these ends are carried to the new variable
                trialbound(a,1) = xx(i,1); %also carry timestamps for start
                trialbound(a,3) = xx(i+1); %and for end
                a = a+1;
            end
        end
    end
    if datatrials.set == 1 || datatrials.set ==2,
        %%
        %%Separate trials / filter data
        %find number of acquisitions between markers; index per trial
        [K,~] = size(trialbound);
        trialdurT = 1:K; trialdurB = 1:K;

        for i = 1:K,
            if datatrials.set==2,
                trialdurB(i) = (trialbound(i,3) - trialbound(i,1)).*30;%trial duration in # of frames
                trialdurT(i) = trialdurB(i).*25;%trial duration in # of body acquisitions
            else
                trialdurT(i) = ((trialbound(i,3) - trialbound(i,1))-1);%trial duration in # of frames
                trialdurB(i) = trialdurT(i)./25;%trial duration in # of body acquisitions
            end 
        end
        %discard intertrial acquisitions
        CR  = sum(trialdurT);
        Cmat = zeros(CR,4);
        j = 1; %FINE UP TIL HERE....
        for i = 1:K,
            if datatrials.set==1,             
                s = trialbound(i,1)+1; %get index of first frame in current trial
                e = trialbound(i,3)-1; %get index of last frame in current tial
            else
                [c s] = min(abs(trialbound(i,1)-alldataN(:,1))); %get as close to the timestamp as possible
                [c e] = min(abs(trialbound(i,3)-alldataN(:,1))); 
            end
            N = length(alldataN(1,:));
            for ii = s:(e-1),
                for k = 1:N,
                    if iscell(alldataN),
                        Cmat(j,k) = str2num(alldataN{ii,k});
                    else
                        Cmat(j,k) = alldataN(ii,k); %take all data within current trial
                    end
                end
                j = j+1;
            end
        end
    end
else
    trialdurB = length(alldataN)./25; ms = [];
    Cmat = alldataN;
end
clear alldata 
%%
%partition into individual joint matrices
CB = floor(sum(trialdurB));
j = zeros(CB,4);
for i = 1:25, %for each joint..
    ij = i;
    for ii = 1:CB, %for each body acquisition
        if iscell(Cmat),
            j(ii,1) = str2num(Cmat{ij,1});
            
            for k = 2:4,
                j(ii,k) = str2num(Cmat{ij,k+1});
            end
        else %only used for ascii output with timestamp for each joint/acquisition
            if length(Cmat(1,:)) <5,
                j(ii,1) = Cmat(ij,1); %timestamp for current joint and acquisition
                j(ii,2:4) = Cmat(ij,2:4); %xyz for current joint and acquisition
            else %in case joint number is included
                j(ii,1) = Cmat(ij,1); %timestamp for current joint and acquisition
                j(ii,2:4) = Cmat(ij,3:5); %xyz for current joint and acquisition
            end
        end
        ij = ij+25;
    end
    Cm.(jx{i}) = j; %write all frame and xyz data for one joint to a field
end
clear Cmat
%%REORDER AND FILTER DATA 
fprintf('\nReordering and filtering data..');  
if exist('trialorder','var') == 1,
    [Cm,ms, trialdurBr] = KFE_reorder(Cm,trialdurB,trialorder,jx); %reorder trial numbering
    trialdurB = trialdurBr;
else
    ms=[];
end
Cm = KFE_filter(Cm,jx); %filter artefacts

%check for scale
SK = Cm.j0(1,3)-Cm.j1(1,3);
if abs(SK)>100,
    for kk = 1:25,
        Cm.(jx{kk}) = Cm.(jx{kk})./1000;
    end
else

end
end

function [cmRo, ms,trialdurBr] = KFE_reorder(Cm, trialdurB, trialorder,jx)
%%standardizes trialordering; imputs are Cm structure, length of the
%%trialdur variable, and the trialorder array, listing the corresponding,
%%collected trial order
g = 1;
ms = []; %missing trials
L = length(trialorder);
trialdurBr = [1:length(trialorder)];
a =1; k=1;
for i = 1:L,
    Cur = find(trialorder == i); %find the index of the i'th trial
    if ~isempty(Cur);
        End = sum(trialdurB(1:Cur)); %find number of acquisitions up to and including current trial
        for j = 1:25,
            [cmRo.(jx{j})(g:((g+trialdurB(Cur))-1),1:4)] = Cm.(jx{j})(((End-trialdurB(Cur))+1):(End),1:4);  %reordering of every joint on the ith trial
%             [Trial.(jx{j})(g:((g+trialdurB(Cur))-2),1:4)] = Cm.(jx{j})(((End-trialdurB(Cur))+1):(End-1),1:4);           
        end
        trialdurBr(k) = trialdurB(Cur); k = k+1; %reorder durations
        g = g+trialdurB(Cur);
    else
        ms(a) = i; %keep track of missing trials
        a = a+1;
    end
end
end

function Cm = KFE_filter(Cm,jx)
for i = [1:24],

    for j = 2:4,
        %apply Savitsky-Golay smoothing filter with span=15 and degree=5
        %note that each dimension (x,y,z) is smoothed separately
        Cm.(jx{i})(:,j) = smooth(Cm.(jx{i})(:,j),15,'sgolay',5); 
    end
end
end
