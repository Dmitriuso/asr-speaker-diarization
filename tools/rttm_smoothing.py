import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RTTM = ROOT / "data/pred_rttms"
file_name = "mondialisation_30min_own"

rttm_file = open(str(RTTM / f"{file_name}.rttm")).readlines()

time_stamps, speakers = [], []

for line in rttm_file:
    speaker = re.findall(r'speaker_\d', line)
    speakers.extend(speaker)
    time_stamp = re.findall(r'\d+\.\d+', line)
    time_stamps.append(time_stamp)

starts = [i[0] for i in time_stamps]
durations = [j[1] for j in time_stamps]

idx_spans = []
for idx in range(len(speakers)):
    if speakers[idx] == speakers[idx-1]:
        idx_spans.append(1)
    else:
        idx_spans.append(0)

indices = [0]
for index, element in enumerate(idx_spans):
    if element == 0:
        indices.append(index)
    else:
        pass

new_starts, new_durations, new_speakers = [], [], []


def add_up_durations(values_list, start, end):
    counter = 0
    for value in values_list[start:end]:
        counter += float(value)
    return counter


for i in range(len(indices)):
    new_starts.append(starts[indices[i]])
    new_speakers.append(speakers[indices[i]])

for k in range(len(indices)-1):
    new_durations.append(add_up_durations(durations, indices[k], indices[k+1]))
new_durations.append(add_up_durations(durations, indices[-1], None))

new_diarization = []
for st, dur, sp in zip(new_starts, new_durations, new_speakers):
    diarization = (f'{float(st): .3f}', f'{float(dur): .3f}', sp)
    new_diarization.append(diarization)

with open(str(RTTM / f"{file_name}_smooth.rttm"), "wt") as f:
    for trio in new_diarization:
        f.write("SPEAKER mondialisation_full 1   " + str(trio[0]) + "   " + str(trio[1]) + " <NA> <NA> " + trio[2] + " <NA> <NA>")
        f.write("\n")

if __name__ == '__main__':
    print(len(starts), len(durations))
    print(len(new_starts), len(new_durations), len(new_speakers))
    print(new_starts)
    print(new_durations)
    print(new_speakers)
    print(new_diarization)
