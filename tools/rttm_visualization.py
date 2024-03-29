from pathlib import Path

from nemo.collections.asr.parts.utils.speaker_utils import (
    labels_to_pyannote_object, rttm_to_labels)

ROOT = Path(__file__).parent.parent
rttm_dir = ROOT / "data/pred_rttms"

mondialisation_rttm = rttm_dir / "Tjl_time_stamps_own_smooth.rttm"

labels = rttm_to_labels(mondialisation_rttm)
reference = labels_to_pyannote_object(labels)


if __name__ == "__main__":
    print(labels)
    reference
