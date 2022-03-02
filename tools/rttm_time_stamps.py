import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
bench_dir = ROOT / "data/radiofrance_bench"
rttm_dir = ROOT / "data/pred_rttms"

bench_file = (
    bench_dir
    / "youtube=p601i1-vTjI/sources/transcript_AMBERSCRIPT_WITH_DIARIZATION.txt"
)


def convert_time_stamp(time_stamp: str):
    parsed = re.findall(r"\d+", time_stamp)
    return int(parsed[0]) * 3600 + int(parsed[1]) * 60 + int(parsed[2])


def extract_time_stamps(file_path, first_pattern, second_pattern):
    list_from_file = open(file_path).readlines()
    speakers = []
    time_stamps = []
    for line in list_from_file:
        speaker_pattern = re.compile(first_pattern)
        speaker = speaker_pattern.search(line)
        time_stamp_pattern = re.compile(second_pattern)
        time_stamp = time_stamp_pattern.search(line)
        if speaker is not None:
            speakers.append(speaker.group())
        else:
            pass
        if time_stamp is not None:
            time_stamps.append(time_stamp.group())
        else:
            pass
    start = []
    duration = []
    for stamp in time_stamps:
        sec_stamp = convert_time_stamp(stamp)
        start.append(sec_stamp)
    for idx in range(len(start) - 1):
        duration.append(start[idx + 1] - start[idx])
    duration.append(0)
    speakers_items = []
    for s in speakers:
        ns = re.sub(r"\s+", "_", s)
        speakers_items.append(ns)
    joint_list = []
    for i, j, k in zip(start, duration, speakers_items):
        trio = (i, j, k)
        joint_list.append(trio)
    print(f"The length of the joint list is {len(joint_list)}")
    # print(start)
    print(f"The length of the start points list is {len(start)}")
    # print(duration)
    print(speakers_items)
    print(f"The length of the durations list is {len(duration)}")
    return joint_list


sonix_time_stamps = extract_time_stamps(
    bench_file, r"(S|s)peaker(\s+)?\d", r"(\[)?\d+\:\d+\:\d+(\])?"
)

with open(str(rttm_dir / "Tjl_time_stamps_bench.rttm"), "wt") as f:
    for time_stamp in sonix_time_stamps:
        f.write(
            "SPEAKER mondialisation_full 1   "
            + str(time_stamp[0])
            + "   "
            + str(time_stamp[1])
            + " <NA> <NA> "
            + time_stamp[2]
            + " <NA> <NA>"
        )
        f.write("\n")
