# -*- coding: utf-8 -*-
"""
Kinematic Feature Extraction
Based on the Matlab scripts of the same name, described in Trujillo et al., 2019
Beh. Res. Meth.
Takes OpenPose-generated (2D) coordinate data and outputs summary tables and figures.
# Runs on Python 3.7 for matplotlib compatibility

Created on Thu Jan 31 11:03:09 2019
Last update: Dec. 07. 2020

@author: James Trujillo
jptrujillo88@hotmail.com
"""


# TODO: come up with a standardized measure - pixels are not helpful for defining any thresholds!
# TODO: add plotting
# TODO: add wrapper and output


# Run check_skeleton(df) for a quick plot of all joints on the first frame



# Where can it find your data?
# OP_dir is the main directory 
OP_dir = "C:/Users/James/Work/SkypeProj/"
# the data directory should contain your motion tracking data
data_dir = OP_dir + "data/"
# annot dir should contain your gesture annotations
# annotations should be exported as .txt files in two steps
# one file should be the exported gesture Type tier, with a _g.txt suffix
# one file should be the exported gesture Hand tier (L,R, B) with a _h.txt suffix
annot_dir = OP_dir + "annotations/"



#import pandas as pd
import statistics
from scipy import signal
import math
import numpy as np
import pandas as pd
import os
os.chdir(OP_dir)
import process_json_data
import matplotlib.pyplot as plt
from collections import Counter


def main(OP_dir, annot_dir, FPS):
    OP_dir = "C:/Users/James/Work/SkypeProj/"

    # first, a loop to go through all the data. Take video name as starting point, match to annotation file
    os.chdir(data_dir)

    #general directory
    dirs = os.listdir(data_dir)

    full_pp = []
    dyad = []
    participant = []
    gest_num_list = []
    gest_type_list = []
    dur_list = []
    hand_list = []
    Lsub_list = []
    Rsub_list = []
    L_rhythm_list = []
    R_rhythm_list = []
    L_peak_nPVI_list = []
    R_peak_nPVI_list = []
    nPVI_peak_list = []
    L_nPVI_list = []
    R_nPVI_list = []
    nPVI_list = []
    vert_height_list = []
    Lmax_list = []
    Rmax_list = []
    Lsize_list = []
    Rsize_list = []
    peakL_list = []
    peakR_list = []
    holdcount_list = []
    holdtime_list = []
    hold_avg_list = []
    vol_list = []
    space_use_L_list = []
    space_use_R_list = []
    mcneillian_maxL_list = []
    mcneillian_maxR_list = []
    mcneillian_modeL_list  =[]
    mcneillian_modeR_list = []

    #dur = df.shape[0]  # number of datapoints
    for Vid in dirs:
        if os.path.isdir(data_dir + Vid):
            # here we need to split the dataframe according to some other data
            annots_dir_g = annot_dir + Vid[0:6] + "_g.txt"
            annot_df_g = pd.read_csv(annots_dir_g, sep="\t")
            annots_dir_h = annot_dir + Vid[0:6] + "_h.txt"
            annot_df_h = pd.read_csv(annots_dir_h, sep="\t")
            #  convert msec to frames
            annot_df_g["BeginF"] = (annot_df_g["Begin Time - msec"] / 1000) * 25
            annot_df_g["EndF"] = (annot_df_g["End Time - msec"] / 1000) * 25
    
            FPS = 25  #TODO: remove this once the wrapper is working 
    
            print(Vid)
            # now loop through each gesture
            for gesture_idx, _ in annot_df_g.iterrows():
                print("\t" + str(gesture_idx))
                firstFrame = math.floor(annot_df_g["BeginF"][gesture_idx])
                lastFrame = math.floor(annot_df_g["EndF"][gesture_idx])
                hand = annot_df_h["Hand"][gesture_idx]
                
                dur = annot_df_g["End Time - msec"][gesture_idx] - annot_df_g["Begin Time - msec"][gesture_idx]
                
                df = process_json_data.main(data_dir + Vid + "/data/", firstFrame, lastFrame)
                df = flip_data(df)
                
                try:
                    #L_subs, R_subs, subslocs_L, subslocs_R = calc_submoves(df, FPS, hand)  # get submovements
                    maxheight = calc_vert_height(df, hand)  # get vertical height
                    Lmax, Rmax = calc_maxSize(df, hand)  # get size from body
                    LSize, RSize, Luse, Ruse = calc_jointSize(df, Lmax, Rmax, hand)  # get number of joints used
                    Lsubs, Rsubs, L_peaks, R_peaks = calc_submoves(df, FPS, hand)  # get number of submovements
                    L_rhythm_stab = calc_rhythm_stability(L_peaks)
                    R_rhythm_stab = calc_rhythm_stability(R_peaks)
                    L_peak_nPVI = calc_peak_nPVI(L_peaks)
                    R_peak_nPVI = calc_peak_nPVI(R_peaks)
                    L_nPVI = calc_nPVI(df['L_Hand'], FPS)
                    R_nPVI = calc_nPVI(df['R_Hand'],FPS)
                        
                    if hand == 'L' or hand == 'B':
                        peakL = calc_peakVel(df['L_Hand'], FPS)  # left hand peak velocity
                    else:
                        peakL = 0
                    if hand == 'R' or hand == 'B':
                        peakR = calc_peakVel(df['R_Hand'], FPS)  # right hand peak velocity
                    else:
                        peakR = 0
                    hold_count, hold_time, hold_avg = calc_holds(df, Ruse, Luse, L_peaks, R_peaks, FPS, hand)  # holds
                    vol = calc_volume_size(df, hand)
                    space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, mcneillian_modeL, mcneillian_modeR = calc_mcneillian_space(df, hand)
    
    
                except:
                    print("missing data!")
                
                    maxheight = 'NA'
                    Lmax = 'NA'
                    Rmax = 'NA'
                    LSize = 'NA'
                    RSize = 'NA'
                    peakL = 'NA'
                    peakR = 'NA'
                    Lsubs = 'NA'
                    Rsubs = 'NA'
                    L_rhythm_stab = 'NA'
                    R_rhythm_stab = 'NA'
                    L_nPVI = 'NA'
                    R_nPVI = 'NA'
                    L_peak_nPVI = 'NA'
                    R_peak_nPVI = 'NA'
                    hold_count = 'NA'
                    hold_time = 'NA'
                    hold_avg = 'NA'
                    vol = 'NA'
                    space_use_L = 'NA'
                    space_use_R = 'NA'
                    mcneillian_modeL = 'NA'
                    mcneillian_modeR = 'NA'
                    mcneillian_maxL = 'NA'
                    mcneillian_maxR = 'NA'
                full_pp.append(Vid)
                dyad.append(Vid[0:3])
                participant.append(Vid[5:6])
                hand_list.append(hand)
                gest_num_list.append(gesture_idx)
                gest_type_list.append(annot_df_g["Type"][gesture_idx])
                dur_list.append(dur)
                Lsub_list.append(Lsubs)
                Rsub_list.append(Rsubs)
                L_rhythm_list.append(L_rhythm_stab)
                R_rhythm_list.append(R_rhythm_stab)
                L_peak_nPVI_list.append(L_peak_nPVI)
                R_peak_nPVI_list.append(R_peak_nPVI)
                L_nPVI_list.append(L_nPVI)
                R_nPVI_list.append(R_nPVI)
                vert_height_list.append(maxheight)
                Lmax_list.append(Lmax)
                Rmax_list.append(Rmax)
                Lsize_list.append(LSize)
                Rsize_list.append(RSize)
                peakL_list.append(peakL)
                peakR_list.append(peakR)
                holdcount_list.append(hold_count)
                holdtime_list.append(hold_time)
                hold_avg_list.append(hold_avg)
                vol_list.append(vol)
                space_use_L_list.append(space_use_L)
                space_use_R_list.append(space_use_R)
                mcneillian_maxL_list.append(mcneillian_maxL)
                mcneillian_maxR_list.append(mcneillian_maxR)
                mcneillian_modeL_list.append(mcneillian_modeL)
                mcneillian_modeR_list.append(mcneillian_modeR)

    df_full = pd.DataFrame(np.column_stack([full_pp, participant, hand_list, 
                                            gest_num_list, gest_type_list, dur_list, 
                                            Lsub_list, Rsub_list, L_rhythm_list, R_rhythm_list, 
                                            L_peak_nPVI_list, R_peak_nPVI_list, L_nPVI_list, R_nPVI_list, 
                                            vert_height_list, Lmax_list, Rmax_list,
                                            Lsize_list, Rsize_list, peakL_list, peakR_list, 
                                            holdcount_list, holdtime_list, hold_avg_list, 
                                            vol_list, space_use_L_list, space_use_R_list,
                                            mcneillian_maxL_list, mcneillian_maxR_list, 
                                            mcneillian_modeL_list,mcneillian_modeR_list]),
                           columns=['file', 'participant', 'hand',
                                    'gesture_num', 'gesture_type', 'duration',
                                    'Lsubs', 'Rsubs', 'L_rhythm', 'R_rhythm',
                                    'L_peak_nPVI','R_peak_nPVI', 'L_nPVI','R_nPVI',
                                    'Vert_Amp', 'Lmax', 'Rmax', 
                                    'Lsize', 'Rsize', 'peakL', 'peakR',
                                    'holdcount', 'holdtime', 'hold_avg', 
                                    'volume', 'MN_spaceL', 'MN_spaceR',
                                    'MN_maxL', 'MN_maxR', 'MN_modeL', 'MN_modeR']
                           )
    
    rhythm = []
    for index,row in df_full.iterrows():
        if df_full['hand'][index] == 'R':
            rhythm.append(df_full['R_rhythm'][index])
            nPVI_list.append(df_full['R_nPVI'][index])
            nPVI_peak_list.append(df_full['R_peak_nPVI'][index])

        elif df_full['hand'][index] == 'L':
            rhythm.append(df_full['L_rhythm'][index])
            nPVI_list.append(df_full['L_nPVI'][index])
            nPVI_peak_list.append(df_full['L_peak_nPVI'][index])


        else:
            if df_full['L_rhythm'][index] != 'NA' and df_full['R_rhythm'][index] != 'NA':
                rhythm.append((float(df_full['L_rhythm'][index]) +float(df_full['R_rhythm'][index]))/2)
                nPVI_list.append((float(df_full['L_nPVI'][index]) +float(df_full['R_nPVI'][index]))/2)
                nPVI_peak_list.append((float(df_full['L_peak_nPVI'][index]) +float(df_full['R_peak_nPVI'][index]))/2)                

            elif df_full['L_rhythm'][index] != 'NA':
                rhythm.append(df_full['L_rhythm'][index])
                nPVI_list.append(df_full['L_nPVI'][index])
                nPVI_peak_list.append(df_full['L_peak_nPVI'][index])


            elif df_full['R_rhythm'][index] != 'NA':
                rhythm.append(df_full['R_rhythm'][index])
                nPVI_list.append(df_full['R_nPVI'][index])
                nPVI_peak_list.append(df_full['R_peak_nPVI'][index])

            else:
                rhythm.append('NA')
                nPVI_list.append('NA')
                nPVI_peak_list.append('NA')

    df_full['rhythm'] = rhythm
    df_full['nPVI'] = nPVI_list
    df_full['nPVI_peak'] = nPVI_peak_list

    df_full.to_csv(OP_dir + "KFE_output.csv", index=False)  


def calc_nPVI(Hand,FPS):
    dist,_ = calculate_distance(Hand, FPS)
        # first get all IOIs
    temp_interval = [dist[idx]-dist[idx-1] for idx in range(1,len(dist))]
    PVI = []
    for idx in range(1,len(temp_interval)):    
         PVI.append((temp_interval[idx]-temp_interval[idx-1])/((temp_interval[idx]+temp_interval[idx-1])/2)) 
    nPVI = np.mean(PVI)*100
    return nPVI


def calc_peak_nPVI(peaks):
    if isinstance(peaks, np.ndarray) and len(peaks) >= 3:
        # first get all IOIs
        temp_interval = [peaks[idx]-peaks[idx-1] for idx in range(1,len(peaks))]
        PVI = []
        for idx in range(1,len(temp_interval)):    
             PVI.append((temp_interval[idx]-temp_interval[idx-1])/((temp_interval[idx]+temp_interval[idx-1])/2)) 
        peak_nPVI = np.mean(PVI)*100
    else:
        peak_nPVI = 'NA'
    return peak_nPVI
      
def calc_rhythm_stability(peaks):
    # higher scores indicate more temporal variability, and thus less stable rhythm
    if isinstance(peaks, np.ndarray) and len(peaks) >= 3:
        # first get temporal interval (in frames) between submovements
        temp_interval = [peaks[idx]-peaks[idx-1] for idx in range(1,len(peaks))]
        # intermittency is the variance of these intervals
        stability = np.std(temp_interval)

    else:
       stability= 'NA'
    return stability


def flip_data(df):
    # first, we flip the y-axis, if needed
    if df['Nose'][0][1] < df['MidHip'][0][1]:
        cols = list(df)
        # first get the vertical height of the configuration
        # we only do this for the first frame; the transformation will be applied to all frames
        maxpoint = []
        for joint in cols:
            maxpoint.append(df[joint][0][1])
        # iterate over each joint, in each frame, to flip the y-axis
        for frame in range(len(df)):
            for joint in cols:
                ytrans = max(maxpoint) - df[joint][frame][1] - 1
                df[joint][frame][1] = ytrans

    return df


def calculate_distance(Hand, FPS):
    """
    This just calculates the displacement between each set of points, then the
    velocity from the displacement.
    """
    IDX = 0
    dist = []
    vel = []
    for coords in Hand[1:]:
        Prev_coords = Hand[IDX]
        # first calculate displacement
        DISPLACE = math.hypot(float(coords[0]) - float(Prev_coords[0]), float(coords[1]) - float(Prev_coords[1]))
        dist.append(DISPLACE)
        # then calculate velocity
        vel.append(DISPLACE * FPS)

        IDX = IDX + 1
    dist = list(dist)
    vel = list(vel)
    return dist, vel


def calc_vert_height(df, hand):
    # Vertical amplitude
    # H: 0 = below midline;
    #    1 = between midline and middle-upper body;
    #    2 = above middle-upper body, but below shoulders;
    #    3 = between shoulders nad middle of face;
    #    4 = between middle of face and top of head;
    #    5 = above head

    H = []
    for index, frame in df.iterrows():
        SP_mid = ((df.loc[index, "Neck"][1] - df.loc[index, "MidHip"][1]) / 2) + df.loc[index, "MidHip"][1]
        Mid_up = ((df.loc[index, "Nose"][1] - df.loc[index, "Neck"][1]) / 2) + df.loc[index, "Neck"][1]
        Eye_mid = (df.loc[index, "REye"][1] + df.loc[index, "LEye"][1]) / 2  # mean of the two eyes vert height
        Head_TP = ((Eye_mid - df.loc[index, "Nose"][1]) * 2) + df.loc[index, "Nose"][1]

        if hand == "B":
            hand_height = max([df.loc[index, "R_Hand"][1], df.loc[index, "L_Hand"][1]])
        else:
            hand_str = hand + "_Hand"
            hand_height = df.loc[index][hand_str][1]

        if hand_height > SP_mid:
            if hand_height > Mid_up:
                if hand_height > df.loc[index, "Neck"][1]:
                    if hand_height > df.loc[index, "Nose"][1] :
                        if hand_height > Head_TP:
                            H.append(5)
                        else:
                            H.append(4)
                    else:
                        H.append(3)
                else:
                    H.append(2)
            else:
                H.append(1)
        else:
            H.append(0)
    MaxHeight = max(H)
    return MaxHeight


def calc_mean_pos(hand, axis, startpos, stoppos):
    # just takes the mean of a set of positional values
    mean_pos = []

    for ax in axis:
        position = []
        for index in range(startpos, stoppos):
            position.append(hand.loc[index][ax])
        mean_pos.append(statistics.mean(position))

    return mean_pos


def calc_maxSize(df, hand):
    # Calculates the maximum distance of each hand from its position at the beginning of the video/input

    Ldis = []
    Rdis = []

    if hand == 'L' or hand == 'B':
        LEo = calc_mean_pos(df['L_Hand'], [0, 1], 0, 4)  # change tuple input to 0,1,2 if using 3D
    if hand == 'R' or hand == 'B':
        REo = calc_mean_pos(df['R_Hand'], [0, 1], 0, 4)  # change tuple input to 0,1,2 if using 3D

    # calculates the distance of the hands, at each frame, from their starting position
    # assumes 2D data
    for index, frame in df.iterrows():
        if index > 4:
            if hand == 'L' or hand == 'B':
                Ldis.append(math.sqrt(((df.loc[index, "L_Hand"][0] - LEo[0]) ** 2) +
                                      ((df.loc[index, "L_Hand"][1] - LEo[1]) ** 2)))
            if hand == 'R' or hand == 'B':
                Rdis.append(math.sqrt(((df.loc[index, "R_Hand"][0] - REo[0]) ** 2) +
                                      ((df.loc[index, "R_Hand"][1] - REo[1]) ** 2)))

    if hand == 'L' or hand == 'B':
        LMax = max(Ldis)
    else:
        LMax = 0
    if hand == 'R' or hand == 'B':
        RMax = max(Rdis)
    else:
        RMax = 0
    return LMax, RMax


def calc_jointSize(df, LMax, RMax, hand):
    # Calculates Size based on number of joints:
    # 2 = hand and elbow used
    # 1 = only hand movement, elbow stationary
    # 0 = no significant movement in that arm

    LdisElb = []
    RdisElb = []
    LElbMove = []
    RElbMove = []
    LSize = 0
    RSize = 0

    # is there significant movement in the hands?
    if LMax > 0.15:  # hand use
        Luse = 1
    else:
        Luse = 0
    if RMax > 0.15:
        Ruse = 1
    else:
        Ruse = 0

    # calculate elbow use
    # first get the elbow origin points
    LElbo = calc_mean_pos(df['LElb'], [0, 1], 0, 4)  # change tuple input to 0,1,2 if using 3D
    RElbo = calc_mean_pos(df['RElb'], [0, 1], 0, 4)  # change tuple input to 0,1,2 if using 3D

    # then calculate distance from starting position at each frame
    # this is used just to get the average amount distance across the whole time window
    for index, frame in df.iterrows():
        if index > 4:
            LdisElb.append(math.sqrt(((df.loc[index, "LElb"][0] - LElbo[0]) ** 2) +
                                     ((df.loc[index, "LElb"][1] - LElbo[1]) ** 2)))
            RdisElb.append(math.sqrt(((df.loc[index, "RElb"][0] - RElbo[0]) ** 2) +
                                     ((df.loc[index, "RElb"][1] - RElbo[1]) ** 2)))
    # now check if the distance is ever greater than the mean distance + 1SD
    for disval in LdisElb:
        LElbMove.append(disval > (statistics.mean(LdisElb) + statistics.stdev(LdisElb)))
    for disval in RdisElb:
        RElbMove.append(disval > (statistics.mean(RdisElb) + statistics.stdev(RdisElb)))
    # if the elbow ever moves beyond this threshold, we say there is elbow use
    if True in (M > 0 for M in LElbMove):
        LElbUse = 1
    else:
        LElbUse = 0
    if True in (M > 0 for M in RElbMove):
        RElbUse = 1
    else:
        RElbUse = 0

    # now determine the overall size
    # Right side
    if Ruse == 1 and RElbUse == 1:
        RSize = 2  # hand and elbow were moving
    elif Ruse == 1 and RElbUse == 0:
        RSize = 1  # only the lower arm
    else:
        RSize = 0  # no arm movement
    # Left side
    if Luse == 1 and LElbUse == 1:
        LSize = 2  # hand and elbow were moving
    elif Luse == 1 and LElbUse == 0:
        RSize = 1  # only the lower arm
    else:
        LSize = 0  # no arm movement

    return LSize, RSize, Luse, Ruse


def calc_submoves(df, FPS, hand):
    # calculates the number of submovements, and gives the indices(locations) of the peaks
    if hand == 'L' or hand == 'B':
        LHdelta, LH_S = calculate_distance(df["L_Hand"], FPS)
        LH_Sm = smooth_moving(LH_S, 3)
        L_peaks, _ = signal.find_peaks(LH_Sm, height=0.2, prominence=0.2, distance=3)
        Lsubs = len(L_peaks)
    else:
        Lsubs = 0
        L_peaks = 0
    if hand == 'R' or hand == 'B':
        RHdelta, RH_S = calculate_distance(df["R_Hand"], FPS)  # displacement (ie delta) between frames
        RH_Sm = smooth_moving(RH_S,2)
        R_peaks, _ = signal.find_peaks(RH_Sm, height=0.2, prominence=0.2, distance=3)
        Rsubs = len(R_peaks)
    else:
        Rsubs = 0
        R_peaks = 0


    return Lsubs, Rsubs, L_peaks, R_peaks


def calc_peakVel(HandArr, FPS):
    # smooths the velocity array and then takes the max value
    # takes one hand array as input

    _, VelArray = calculate_distance(HandArr, FPS)
    HandVel_Sm = smooth_moving(VelArray, 3)

    return max(HandVel_Sm)


def smooth_moving(data, degree):
    # uses a moving window to smooth an array

    triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
    smoothed = []

    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point) / np.sum(triangle))
    # Handle boundaries
    smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def find_movepauses(velocity_array):
    # finds moments when velocity is below a particular threshold
    # returns array of indices for those moments

    pause_ix = []
    for index, velpoint in enumerate(velocity_array):

        if velpoint < 200: # 200px/s
            pause_ix.append(index)
    if len(pause_ix) == 0:
        pause_ix = 0

    return pause_ix


def calc_holds(df, Ruse, Luse, subslocs_L, subslocs_R, FPS, hand):
    # calculates the number of holds, time spent in a hold, and the average duration of any holds
    
    
    if Ruse == 1:

        # elbow
        _, RE_S = calculate_distance(df["RElb"], FPS)  # R elbow velocity
        GERix = find_movepauses(RE_S)
        # hand
        _, RH_S = calculate_distance(df["R_Hand"], FPS)
        GRix = find_movepauses(RH_S)
        # finger
        _, RF_S = calculate_distance(df["R_finger"], FPS)
        GFRix = find_movepauses(RF_S)

        # now find holds for the entire right side
        GR = []
        if isinstance(GERix, list) and isinstance(GRix, list) and isinstance(GFRix, list):
            for handhold in GRix:
                for elbowhold in GERix:
                    for fingerhold in GFRix:
                        if handhold == elbowhold:
                            if elbowhold == fingerhold:
                                GR.append(handhold)  # this is all holds of the entire right side

    if Luse == 1:

        # elbow
        _, LE_S = calculate_distance(df["LElb"], FPS)  # L elbow velocity
        GELix = find_movepauses(LE_S)
        # hand
        _, LH_S = calculate_distance(df["L_Hand"], FPS)
        GLix = find_movepauses(LH_S)
        # finger
        _, LF_S = calculate_distance(df["L_finger"], FPS)
        GFLix = find_movepauses(LF_S)

        # now find holds for the entire right side
        if isinstance(GELix, list) and isinstance(GLix, list) and isinstance(GFLix, list):
            GL = []
            for handhold in GLix:
                for elbowhold in GELix:
                    for fingerhold in GFLix:
                        if handhold == elbowhold:
                            if elbowhold == fingerhold:
                                GL.append(handhold)  # this is all holds of the entire right side

    if (hand == 'B' and 'GL' in locals() and 'GR' in locals()) or \
            (hand == 'L' and 'GL' in locals()) or (hand == 'R' and 'GR' in locals()):
        # find holds involving both hands
        full_hold = []
        if hand == 'B' and Ruse == 1 and Luse == 1:
            for left_hold in GL:  # check, for each left hold,
                for right_hold in GR:  # if there is a corresponding right hold
                    if left_hold == right_hold:
                        full_hold.append(left_hold)  # this is the time position of the hold
        elif hand == 'L' and Luse == 1:
            full_hold = GL
        elif hand == 'R' and Ruse == 1:
            full_hold = GR

        # now we need to cluster them together
        if len(full_hold) > 0:
            #full_hold = [9, 13, 14, 15, 19]
            hold_cluster = [[full_hold[0]]]
            clustercount = 0
            holdcount = 1
            for idx in range(1, len(full_hold)):
                # if the next element of the full_hold list is not equal to the previous value,
                if full_hold[idx] != hold_cluster[clustercount][holdcount - 1] + 1:
                    clustercount += 1
                    holdcount = 1
                    hold_cluster.append([full_hold[idx]])  # then start a new cluster
                else:  # otherwise add the current hold to the current cluster
                    hold_cluster[clustercount].append(full_hold[idx])
                    holdcount += 1

            # we don't want holds occuring at the very beginning or end of an analysis segment
            # so we define these points as the first and last submovement, and remove all holds
            # outside these boundaries
            if hand == 'B':
                if len(subslocs_L) > 0 and len(subslocs_R) > 0:
                    initial_move = min([min(subslocs_L), min(subslocs_R)])
                else:
                    initial_move = hold_cluster[0][0]
            elif hand == 'L':
                if len(subslocs_L) > 0:
                    initial_move = min(subslocs_L)
                else:
                    initial_move = hold_cluster[0][0]
            elif hand == 'R':
                if len(subslocs_R) > 0:
                    initial_move = min(subslocs_R)
                else:
                    initial_move = hold_cluster[0][0]

            #final_move = max([subslocs_L, subslocs_R])

            for index in range(0, len(hold_cluster)):
                if hold_cluster[0][0] < initial_move:
                    hold_cluster.pop(0)
            #for index in range(len(hold_cluster), 0, -1): # do not remove final holds -- this is because retractions have now been removed from annotations (12-5-2020)
            #    if hold_cluster[len(hold_cluster)][0] > final_move:
            #        hold_cluster.pop(len(hold_cluster))

            # now for the summary stats: find the total hold time
            hold_count = 0
            hold_time = 0
            hold_avg = []

            for index in range(0, len(hold_cluster)):
                if len(hold_cluster[index]) >= 3:
                    hold_count += 1  # total number of single holds
                    hold_time += len(hold_cluster[index])  # get the number of frames
                    hold_avg.append(len(hold_cluster[index])/FPS)  # used to calculate average holdtime

            hold_time /= FPS  # divide by FPS to get actual time
            if len(hold_avg) > 0:
                hold_avg = statistics.mean(hold_avg)
            else:
                hold_avg = 0

            return hold_count, hold_time, hold_avg

        else:  # if no full holds were found, return 0s
            hold_count = 0
            hold_time = 0
            hold_avg = 0
            return hold_count, hold_time, hold_avg
    else:
        hold_count = 0
        hold_time = 0
        hold_avg = 0
        return hold_count, hold_time, hold_avg



def calc_volume_size(df, hand):
    # calculates the volumetric size of the gesture, ie how much visual space was utlized by the hands
    # for 3D data, this is actual volume (ie. using z-axis), for 2D this is area, using only x and y\

    if hand == 'B':
        x_max = max([df['R_finger'][0][0], df['L_finger'][0][0]])
        x_min = min([df['R_finger'][0][0], df['L_finger'][0][0]])
        y_max = max([df['R_finger'][0][1], df['L_finger'][0][1]])
        y_min = min([df['R_finger'][0][1], df['L_finger'][0][1]])
    else:
        hand_str = hand + '_finger'
        x_min = df[hand_str][0][0]
        x_max = df[hand_str][0][0]
        y_min = df[hand_str][0][1]
        y_max = df[hand_str][0][1]

    if len(df['R_finger'][0]) > 2:
        if hand == 'B':
            x_max = max([df['R_finger'][0][2], df['L_finger'][0][2]])
            x_min = min([df['R_finger'][0][2], df['L_finger'][0][2]])
        else:
            z_min = df[hand_str][0][2]
            z_max = df[hand_str][0][2]
    # at each frame, compare the current min and max with the previous, to ultimately find the outer values
    if hand == 'B':
        hand_list = ['R_finger', 'L_finger']
    else:
        hand_list = [hand_str]

    for frame in range(1, len(df)):
        for hand_idx in hand_list:
            if df[hand_idx][frame][0] < x_min:
                x_min = df[hand_idx][frame][0]
            if df[hand_idx][frame][0] > x_max:
                x_max = df[hand_idx][frame][0]
            if df[hand_idx][frame][0] < y_min:
                y_min = df[hand_idx][frame][1]
            if df[hand_idx][frame][1] > y_max:
                y_max = df[hand_idx][frame][1]
            if len(df[hand_idx][0]) > 2:
                if df[hand_idx][frame][2] < z_min:
                    z_min = df[hand_idx][frame][2]
                if df[hand_idx][frame][2] > z_max:
                    z_max = df[hand_idx][frame][2]

    if len(df['R_finger'][0]) > 2:
        # get range
        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min
        # get volume
        vol = x_len * y_len * z_len
    else:
        x_len = x_max - x_min
        y_len = y_max - y_min
        # get area (ie volume)
        vol = x_len * y_len
    return vol


def calc_mcneillian_space(df, hand_ind):
    # this calls the define_mcneillian_grid function for each frame, then assign the hand to one space for each frame
    # output:
    # space_use - how many unique spaces were traversed
    # mcneillian_max - outer-most main space entered
    # mcneillian_mode - which main space was primarily used
    # 1 = Center-center
    # 2 = Center
    # 3 = Periphery
    # 4 = Extra-Periphery
    # subsections for periphery and extra periphery:
    # 1 = upper right
    # 2 = right
    # 3 = lower right
    # 4 = lower
    # 5 = lower left
    # 6 = left
    # 7 = upper left
    # 8 = upper
    if hand_ind == 'B':
        hands = ['L_Hand','R_Hand']
    else:
        hands = [hand_ind + '_Hand']
    # compare, at each frame, each hand to the (sub)section limits, going from inner to outer, clockwise
    for hand in hands:
        Space = []

        for frame in range(len(df)):

            cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax = \
            define_mcneillian_grid(df, frame)

            if cc_xmin < df[hand][frame][0] < cc_xmax and cc_ymin < df[hand][frame][1] < cc_ymax:
                Space.append(1)
            elif c_xmin < df[hand][frame][0] < c_xmax and c_ymin < df[hand][frame][1] < c_ymax:
                Space.append(2)
            elif p_xmin < df[hand][frame][0] < p_xmax and p_ymin < df[hand][frame][1] < p_ymax:
                # if it's in the periphery, we need to also get the subsection
                # first, is it on the right side?
                if cc_xmax < df[hand][frame][0]:
                    # if so, we narrow down the y location
                    if cc_ymax < df[hand][frame][1]:
                        Space.append(31)
                    elif cc_ymin < df[hand][frame][1]:
                        Space.append(32)
                    else:
                        Space.append(33)
                elif cc_xmin < df[hand][frame][0]:
                    if c_ymax < df[hand][frame][1]:
                        Space.append(38)
                    else:
                        Space.append(34)
                else:
                    if cc_ymax < df[hand][frame][1]:
                        Space.append(37)
                    elif cc_ymin < df[hand][frame][1]:
                        Space.append(36)
                    else:
                        Space.append(35)
            else:  # if it's not periphery, it has to be extra periphery. We just need to get subsections
                if c_xmax < df[hand][frame][0]:
                    if cc_ymax < df[hand][frame][1]:
                        Space.append(41)
                    elif cc_ymin < df[hand][frame][1]:
                        Space.append(42)
                    else:
                        Space.append(43)
                elif cc_xmin < df[hand][frame][0]:
                    if c_ymax < df[hand][frame][1]:
                        Space.append(48)
                    else:
                        Space.append(44)
                else:
                    if c_ymax < df[hand][frame][1]:
                        Space.append(47)
                    elif c_ymin < df[hand][frame][1]:
                        Space.append(46)
                    else:
                        Space.append(45)
        if hand == 'L_Hand':
            Space_L = Space
        else:
            Space_R = Space

    # how many spaces used?
    if hand_ind == 'L' or hand_ind == 'B':
        space_use_L = len(set(Space_L))
        if max(Space_L) > 40:
            mcneillian_maxL = 4
        elif max(Space_L) > 30:
            mcneillian_maxL = 3
        else:
            mcneillian_maxL = max(Space_L)
        # which main space was most used?
        mcneillian_modeL = get_mcneillian_mode(Space_L)
    else:
        space_use_L = 0
        mcneillian_maxL = 0
        mcneillian_modeL = 0

    if hand_ind == 'R' or hand_ind == 'B':
        space_use_R = len(set(Space_R))
        # maximum distance (main spaces)
        if max(Space_R) > 40:
            mcneillian_maxR = 4
        elif max(Space_R) > 30:
            mcneillian_maxR = 3
        else:
            mcneillian_maxR = max(Space_R)
        # which main space was most used?
        mcneillian_modeR = get_mcneillian_mode(Space_R)
    else:
        space_use_R = 0
        mcneillian_maxR = 0
        mcneillian_modeR = 0

    return space_use_L, space_use_R, mcneillian_maxL, mcneillian_maxR, mcneillian_modeL, mcneillian_modeR


def get_mcneillian_mode(spaces):
    mainspace = []
    for space in spaces:
        if space > 40:
            mainspace.append(4)
        elif space > 30:
            mainspace.append(3)
        else:
            mainspace.append(space)
    try:
        mcneillian_mode = statistics.mode(mainspace)
    except:
        c = Counter(mainspace)
        mode_count = max(c.values())
        mode_ties = {key for key, count in c.items() if count == mode_count}
        mcneillian_mode = list(mode_ties)[0]
    return mcneillian_mode

def define_mcneillian_grid(df, frame):
    # define the grid based on a single frame, output xmin,xmax, ymin, ymax for each main section
    # subsections can all be found based on these boundaries
    bodycent = df['Neck'][frame][1] - (df['Neck'][frame][1] - df['MidHip'][frame][1])/2
    face_width = (df['LEye'][frame][0] - df['REye'][frame][0])*2
    body_width = df['LHip'][frame][0] - df['RHip'][frame][0]

    # define boundaries for center-center
    cc_xmin = df['RHip'][frame][0]
    cc_xmax = df['LHip'][frame][0]
    cc_len = cc_xmax - cc_xmin
    cc_ymin = bodycent - cc_len/2
    cc_ymax = bodycent + cc_len/2

    # define boundaries for center
    c_xmin = df['RHip'][frame][0] - body_width/2
    c_xmax = df['LHip'][frame][0] + body_width/2
    c_len = c_xmax - c_xmin
    c_ymin = bodycent - c_len/2
    c_ymax = bodycent + c_len/2

    # define boundaries of periphery
    p_ymax = df['LEye'][frame][1] + (df['LEye'][frame][1] - df['Nose'][frame][1])
    p_ymin = bodycent - (p_ymax - bodycent) # make the box symmetrical around the body center
    p_xmin = c_xmin - face_width
    p_xmax = c_xmax + face_width

    return  cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax


def check_skeleton(df):
    cols = list(df)

    for joint in cols:
        plt.scatter(df[joint][0][0], df[joint][0][1])
        plt.text(df[joint][0][0], df[joint][0][1], joint)

    plt.show()


def plot_mcneillian_grid(cc_xmin, cc_xmax, cc_ymin, cc_ymax, c_xmin, c_xmax, c_ymin, c_ymax, p_xmin, p_xmax, p_ymin, p_ymax):

    plt.plot([cc_xmin, cc_xmin], [cc_ymin, cc_ymax])

    plt.plot([cc_xmax, cc_xmax], [cc_ymin, cc_ymax])

    plt.plot([cc_xmin, cc_xmax], [cc_ymin, cc_ymin])

    plt.plot([cc_xmin, cc_xmax], [cc_ymax, cc_ymax])



    plt.plot([c_xmin, c_xmin], [c_ymin, c_ymax])

    plt.plot([c_xmax, c_xmax], [c_ymin, c_ymax])

    plt.plot([c_xmin, c_xmax], [c_ymin, c_ymin])

    plt.plot([c_xmin, c_xmax], [c_ymax, c_ymax])



    plt.plot([p_xmin, p_xmin], [p_ymin, p_ymax])

    plt.plot([p_xmax, p_xmax], [p_ymin, p_ymax])

    plt.plot([p_xmin, p_xmax], [p_ymin, p_ymin])

    plt.plot([p_xmin, p_xmax], [p_ymax, p_ymax])


def get_current_phase(phase_timing_list, Vid):

    phase_timing = phase_timing_list[phase_timing_list['Dyad'] == Vid[0:3]]

    phase_start = []
    phase_end = []

    for index in range(2, 22, 2):
        start_time_full = phase_timing.iloc[0, index]
        start_time_parts = start_time_full.split(":")
        start_time = int(start_time_parts[1]) * 60 + int(start_time_parts[2]) + float("." + start_time_parts[3])
        end_time_full = phase_timing.iloc[0, index + 1]
        end_time_parts = end_time_full.split(":")
        end_time = int(end_time_parts[1]) * 60 + int(end_time_parts[2]) + float("." + end_time_parts[3])

        phase_start.append(start_time)
        phase_end.append(end_time)

    current_phases = pd.DataFrame(np.column_stack([phase_start, phase_end]), columns=['phase_start', 'phase_end'])
    return current_phases


def check_phase(current_phases, start_time):
    start_time_s = start_time/1000

    if start_time_s > current_phases["phase_start"][0] and start_time_s < current_phases["phase_end"][0]:
        phase = 1
    elif start_time_s > current_phases["phase_start"][1] and start_time_s < current_phases["phase_end"][1]:
        phase = 2
    elif start_time_s > current_phases["phase_start"][2] and start_time_s < current_phases["phase_end"][2]:
        phase = 3
    elif start_time_s > current_phases["phase_start"][3] and start_time_s < current_phases["phase_end"][3]:
        phase = 4
    elif start_time_s > current_phases["phase_start"][4] and start_time_s < current_phases["phase_end"][4]:
        phase = 5
    elif start_time_s > current_phases["phase_start"][5] and start_time_s < current_phases["phase_end"][5]:
        phase = 6
    elif start_time_s > current_phases["phase_start"][6] and start_time_s < current_phases["phase_end"][6]:
        phase = 7
    elif start_time_s > current_phases["phase_start"][7] and start_time_s < current_phases["phase_end"][7]:
        phase = 8
    elif start_time_s > current_phases["phase_start"][8] and start_time_s < current_phases["phase_end"][8]:
        phase = 9
    elif start_time_s > current_phases["phase_start"][9] and start_time_s < current_phases["phase_end"][9]:
        phase = 10
    else:
        phase = 0
    return phase


# make sure this is not run when imported
# if __name__ == "__main__":
#    import sys
    #df = main()