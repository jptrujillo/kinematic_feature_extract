%%Wrapper file for Kinematic Feature Extraction routine. Change the paths
%%and input variables to cycle multiple data files through the analysis
%%function. Output is individual folders for each participant/file with
%%.mat file containing the 'skeleton' files for each individual trial, plus
%%figures for velocity profiles and hold-times for each trial. Each folder
%%also contains a text file with the resulting kinematic features.Requires
%%a path where data is stored, which is also used for output folders, and
%%assumes randomized trial order which is then re-organized to a standard
%%numbering. Onsets should be given in number of seconds, starting from
%%time sync (beginning of recording).Plots show velocity profile per hand,
%%per trial (if specified) and indicate submovement peaks with red x's, and
%%holds with vertical bars and numbering
%%by J.P. Trujillo, November 2015. 
%%version 2.1

%%
%%USER INPUT REQUIRED
%path where your kinematic data is stored, as well as trial ordering, if used
Path = 'D:\data\kinect\Prod\'; 

%How are trials/sections-to-be-analyzed defined? 0 = analyze per datafile;
%1 = in datafile; 2 = separate file of durations
datatrials.set = 1; 
%Define trial boundary marks as they appear in data
datatrials.Sbound = 'start'; %leave blank if not using in-data markers
datatrials.Ebound = 'end';
%should the script re-order the trials? Set to 0 for standard (fixed) trial order
%for re-ordering, file should be under Path, with filename format
%'sbj_x_orer.txt', where x is the subject number
datatrials.fixed = 1;
%if using separate onsets/offsets file for trials, give the path and
%subject-specific filename
%order with one row per trial. column 1 = onset, column 2 = offset
datatrials.onoffpath = 'D:\data\kinect\Prod';

%does your data use headers? 
datatrials.header = 0;

%set to 1 for the extra video plot with skeleton + velocity profiles
%(automatically changes FPS to 25 to be compatible with video)
datatrials.extra = 1;

%set for standard Kinect, change if fps differs - note: using extra plot will
%override this!
FPS = 30;  

 %vector for all participant or data file numbers
sbj = [3,9:41];

%%
%Run analyses
KFE_analyze(Path,sbj,datatrials,FPS);
