function newfile = KFE_restruct_file(oldfile)
fprintf('\nRestructuring file..');  
idxS = strfind(oldfile,'start'); %find all start occurences
idxE = strfind(oldfile,'end'); %find all end occurences
KS = find(~cellfun(@isempty,idxS)); %list actual indices
KE = find(~cellfun(@isempty,idxE));

for i = 2:6,
    if length(oldfile{i}) <3, %discard data points before first full record
        SP = i-1;
    end
end    
i = 1;
j = SP;
jj =j+4;
K = ((length(oldfile(SP:end)))+((length(KS)+length(KE)).*4)); %number of rows for new matrix
if rem(K,5) == 0,
    k = K./5;
else
    k = (K-(rem(K,5)))./5;
end
newfile = cell(k,5);

while jj < length(oldfile),

    if ~isempty(find(j==KS)),
        [newfile{i,1}] = 'start';
        j = j+1; jj = jj+1;
    elseif ~isempty(find(j==KE)),
        [newfile{i,1}] = 'end';
        j = j+1; jj = jj+1;
    else
    [newfile{i,1:5}] = oldfile{j:jj};
    j = j+5; jj = jj+5;
    end
    i = i+1;
end
end
 