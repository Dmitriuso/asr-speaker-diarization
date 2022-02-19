import json
import wget

from pathlib import Path
from omegaconf import OmegaConf


ROOT = Path(__file__).parent
data_dir = ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

AUDIO_FILENAME = str(data_dir / 'mondialisation_full.wav')

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization_with_asr.yaml"

if not (data_dir / 'offline_diarization_with_asr.yaml').exists():
    CONFIG = wget.download(CONFIG_URL, data_dir)
else:
    CONFIG = data_dir / 'offline_diarization_with_asr.yaml'

cfg = OmegaConf.load(CONFIG)
print(OmegaConf.to_yaml(cfg))

meta = {
    'audio_filepath': AUDIO_FILENAME,
    'offset': 0,
    'duration':None,
    'label': 'infer',
    'text': '-',
    'num_speakers': None,
    'rttm_filepath': None,
    'uem_filepath' : None
}


if __name__ == '__main__':
    with open(str(data_dir / 'input_manifest.json'), 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')
