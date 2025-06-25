import glob
import os
import re

import mne
import numpy as np
import pyxdf

subject = input("VP: ")

folders = [f for f in os.listdir() if re.search(r'(WaS|ReStoWa|ReBaWa).*' + subject + '.*', f)]

if len(folders) > 0:
    sFolder = folders[0]
else:
    print("No folders found for " + subject)
    input("\nPress enter to quit...")

# set exp for grail file format
if "ReStoWa" in sFolder:
    exp = 1
elif "WaS" in sFolder:
    exp = 2
else:
    exp = 0

xdfFilenames = glob.glob(os.path.join(sFolder, "*xdf"))
eegFilenames1 = glob.glob(os.path.join(sFolder, "*vhdr"))
eegFilenames2 = glob.glob(os.path.join(sFolder, "*bdf"))
edfFilenames = glob.glob(os.path.join(sFolder, "*edf"))
if exp == 1:
    grailFilenames = glob.glob(os.path.join(sFolder, "grail", "*low_low*trigger*")) + glob.glob(
        os.path.join(sFolder, "grail", "*low_high*trigger*")) + glob.glob(
        os.path.join(sFolder, "grail", "*high_low*trigger*")) + glob.glob(
        os.path.join(sFolder, "grail", "*high_high*trigger*"))
elif exp == 2:
    grailFilenames = glob.glob(os.path.join(sFolder, "*stand*trigger*")) + glob.glob(
        os.path.join(sFolder, "*walk*trigger*"))
else:
    grailFilenames = []

#######
# xdf
#######
for x in range(len(xdfFilenames)):
    print("\nChecking " + os.path.split(xdfFilenames[x])[-1] + "...")
    xdfData, xdfHeader = pyxdf.load_xdf(xdfFilenames[x])

    eegTriggers = np.array([])
    eegTimes = np.array([])
    diodeData = np.array([])
    diodeTimes = np.array([])
    pupilLabsEventTimes = np.array([])
    pupilLabsGazeTimes = np.array([])
    xsensTimes = np.array([])
    for stream in xdfData:
        name = stream["info"]["name"][0]
        if len(re.findall(".*LiveAmp.*TriggerIn.*", name)) > 0:
            eegTriggers = np.array(stream["time_series"], dtype=np.int32).flatten()
            eegTimes = np.array(stream["time_stamps"]).flatten()
        if len(re.findall(".*Photo.*", name)) > 0:
            diodeData = np.array(stream["time_series"], dtype=np.int32).flatten()
            diodeTimes = np.array(stream["time_stamps"]).flatten()
        if len(re.findall(".*pupil_labs_Event.*", name)) > 0:
            pupilLabsEventTimes = np.array(stream["time_stamps"]).flatten()
        if len(re.findall(".*pupil_labs_Gaze.*", name)) > 0:
            pupilLabsGazeTimes = np.array(stream["time_stamps"]).flatten()
        if len(re.findall(".*EulerDatagram.*", name)) > 0:
            xsensTimes = np.array(stream["time_stamps"]).flatten()

    if eegTriggers.size == 0:
        print("No EEG trigger data found")
    else:
        triggerCounts = np.zeros(shape=(8), dtype=np.int32)
        triggerBlocks = []
        buttonLTimes = []
        buttonRTimes = []
        blockStart = None
        # stim, diode, left, right, duration, longest gap left, longest gap right
        triggerData = [0, 0, 0, 0, 0, 0, 0]
        for t in range(eegTriggers.size):
            binTrigger = np.binary_repr(eegTriggers[t], width=8)
            for b in range(len(binTrigger)):
                if binTrigger[-1 - b] == "1":
                    triggerCounts[b] += 1

            if binTrigger[-4] == "1":
                # start
                blockStart = t
                triggerData = [0, 0, 0, 0, 0, 0, 0]
                buttonLTimes = [eegTimes[t]]
                buttonRTimes = [eegTimes[t]]
            if binTrigger[-1] == "1":
                # left
                triggerData[2] += 1
                buttonLTimes.append(eegTimes[t])
            if binTrigger[-2] == "1":
                # right
                triggerData[3] += 1
                buttonRTimes.append(eegTimes[t])
            if binTrigger[-3] == "1":
                # stim
                triggerData[0] += 1
            if binTrigger[0] == "1":
                # diode
                triggerData[1] += 1
            if binTrigger[-5] == "1":
                # end
                if not (blockStart is None):
                    triggerData[4] = np.round(eegTimes[t] - eegTimes[blockStart], 3)
                    buttonLTimes.append(eegTimes[t])
                    buttonRTimes.append(eegTimes[t])
                    triggerData[5] = np.round(np.max(np.diff(np.array(buttonLTimes))), 3)
                    triggerData[6] = np.round(np.max(np.diff(np.array(buttonRTimes))), 3)

                    triggerBlocks.append(triggerData)
                    blockStart = None

        print("EEG data:")

        print("Total triggers: " + str(triggerCounts[3]) + " start, " + str(triggerCounts[4]) + " end, " + str(
            triggerCounts[2]) + " stim, " + str(triggerCounts[7]) + " diode, " + str(
            triggerCounts[0]) + " resp left, " + str(triggerCounts[1]) + " resp right")

        print(str(len(triggerBlocks)) + " blocks:")
        for i in range(len(triggerBlocks)):
            print(
                str(i + 1) + ":\t" + str(triggerBlocks[i][0]) + " stim, " + str(triggerBlocks[i][1]) + " diode, " + str(
                    triggerBlocks[i][2]) + " left, " + str(triggerBlocks[i][3]) + " right"
                + "\t|\t" + str(triggerBlocks[i][4]) + "s duration, " + str(
                    triggerBlocks[i][5]) + "s button gap left, " + str(triggerBlocks[i][6]) + "s button gap right")

    if pupilLabsEventTimes.size == 0 or pupilLabsGazeTimes.size == 0:
        print("No Pupil Labs data found")
    else:
        gazeDur = pupilLabsGazeTimes[-1] - pupilLabsGazeTimes[0]
        print("Pupil Labs gaze: " + str(np.round(gazeDur / 60, 2)) + "min duration")
        gapIdx = np.nonzero(np.diff(pupilLabsEventTimes) > 65)[0]
        print("Pupil Labs event: " + str(gapIdx.size) + " gaps found")

    if xsensTimes.size == 0:
        print("No Xsens data found")
    else:
        xsensDur = xsensTimes[-1] - xsensTimes[0]
        print("Xsens data: " + str(np.round(xsensDur / 60, 2)) + "min duration")

    if diodeTimes.size == 0:
        print("No photo sensor data found")
    else:
        diodeDur = diodeTimes[-1] - diodeTimes[0]
        diodeMin = np.min(diodeData)
        diodeMax = np.max(diodeData)
        zeroPerc = np.nonzero(diodeData == 0)[0].size / diodeData.size
        print("Photo sensor: " + str(np.round(diodeDur / 60, 2)) + "min duration, Max "
              + str(diodeMax) + ", Min " + str(diodeMin) + ", " + str(np.round(100 * zeroPerc, 2)) + "% missing data")

#######
# eeg - brainvision
#######
for e in range(len(eegFilenames1)):
    print("\nChecking " + os.path.split(eegFilenames1[e])[-1] + "...")
    eegData = mne.io.read_raw_brainvision(eegFilenames1[e], preload=True, verbose='ERROR')
    events, events_mapping = mne.events_from_annotations(eegData, event_id='auto', verbose='ERROR')
    evt = np.array(events, dtype=np.int32)

    # triggers: begin, end, stim, diode, left, right
    codes = np.zeros(shape=6, dtype=np.int32)
    for key in events_mapping:
        if len(re.findall(".*Begin.*", key)):
            codes[0] = events_mapping[key]
        elif len(re.findall(".*End.*", key)):
            codes[1] = events_mapping[key]
        elif len(re.findall(".*Stim.*", key)):
            codes[2] = events_mapping[key]
        elif len(re.findall(".*Diode.*", key)):
            codes[3] = events_mapping[key]
        elif len(re.findall(".*Left.*", key)):
            codes[4] = events_mapping[key]
        elif len(re.findall(".*Right.*", key)):
            codes[5] = events_mapping[key]

    triggerBlocks = []
    blockStart = None
    for i in range(evt.shape[0]):
        if evt[i, 2] == codes[0]:
            blockStart = i
        if evt[i, 2] == codes[1]:
            if not (blockStart is None):
                # stim, diode, left, right, duration, longest gap left, longest gap right
                triggerData = [0, 0, 0, 0, 0, 0, 0]
                for c in range(4):
                    triggerData[c] = np.nonzero(evt[blockStart:i, 2] == codes[c + 2])[0].size

                triggerData[4] = (evt[i, 0] - evt[blockStart, 0]) * 2 / 1000

                triggerLeft = np.nonzero(evt[blockStart:i, 2] == codes[4])[0] + blockStart
                if triggerLeft.size > 0:
                    tmpIdx = np.concatenate((np.array([blockStart]), triggerLeft, np.array([i])))
                    triggerData[5] = np.amax(np.diff(evt[tmpIdx, 0])) * 2 / 1000
                    # print(np.mean(np.diff(evt[tmpIdx, 0])) * 2 / 1000)
                triggerRight = np.nonzero(evt[blockStart:i, 2] == codes[5])[0] + blockStart
                if triggerRight.size > 0:
                    tmpIdx = np.concatenate((np.array([blockStart]), triggerRight, np.array([i])))
                    triggerData[6] = np.amax(np.diff(evt[tmpIdx, 0])) * 2 / 1000
                    # print(np.mean(np.diff(evt[tmpIdx, 0])) * 2 / 1000)

                triggerBlocks.append(triggerData)
                blockStart = None

    print("Total triggers: " + str(np.nonzero(evt[:, 2] == codes[0])[0].size) + " start, " + str(
        np.nonzero(evt[:, 2] == codes[1])[0].size) + " end, " + str(
        np.nonzero(evt[:, 2] == codes[2])[0].size) + " stim, " + str(
        np.nonzero(evt[:, 2] == codes[3])[0].size) + " diode, " + str(
        np.nonzero(evt[:, 2] == codes[4])[0].size) + " resp left, " + str(
        np.nonzero(evt[:, 2] == codes[5])[0].size) + " resp right")

    print(str(len(triggerBlocks)) + " blocks:")
    for i in range(len(triggerBlocks)):
        print(str(i + 1) + ":\t" + str(triggerBlocks[i][0]) + " stim, " + str(triggerBlocks[i][1]) + " diode, " + str(
            triggerBlocks[i][2]) + " left, " + str(triggerBlocks[i][3]) + " right"
              + "\t|\t" + str(triggerBlocks[i][4]) + "s duration, " + str(
            triggerBlocks[i][5]) + "s button gap left, " + str(triggerBlocks[i][6]) + "s button gap right")

#######
# eeg - bdf
#######
for e in range(len(eegFilenames2)):
    print("\nChecking " + os.path.split(eegFilenames2[e])[-1] + "...")
    eegData = mne.io.read_raw_bdf(eegFilenames2[e], preload=True, verbose='ERROR')

    eegDur = eegData.times[-1] - eegData.times[0]
    print(str(np.round(eegDur / 60, 2)) + "min duration")

#######
# edf
#######
for e in range(len(edfFilenames)):
    print("\nChecking " + os.path.split(edfFilenames[e])[-1] + "...")
    edfData = mne.io.read_raw_edf(edfFilenames[e], preload=True, verbose='ERROR')

    edfDur = edfData.times[-1] - edfData.times[0]
    print(str(np.round(edfDur / 60, 2)) + "min duration")

#######
# grail
#######
if len(grailFilenames) > 0 and exp == 1:
    print("\nChecking " + str(len(grailFilenames)) + " GRAIL files...")
    totalAcc = 0
    totalTrials = 0
    for g in range(len(grailFilenames)):
        grailData = np.loadtxt(grailFilenames[g], skiprows=1)
        accData = np.array(grailData[:, -4:], dtype=np.int32)
        trials = np.unique(accData[np.nonzero(accData[:, 0] > 0)[0], 0])

        accCount = 0
        for t in range(len(trials)):
            trialData = accData[np.nonzero(accData[:, 0] == trials[t])[0], :]
            if trialData[-1, -2] == trialData[-1, -1]:
                accCount += 1

        totalAcc += accCount
        totalTrials += len(trials)

    print("Accuracy: " + str(np.round(100 * totalAcc / totalTrials, 2)) + "%")

input("\nPress enter to quit...")
