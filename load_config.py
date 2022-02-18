import os
import json
import wget

from omegaconf import OmegaConf


ROOT = os.getcwd()
data_dir = os.path.join(ROOT, 'data')
os.makedirs(data_dir, exist_ok=True)

AUDIO_FILENAME = f'{data_dir}/mondialisation_full.wav'

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization_with_asr.yaml"

if not os.path.exists(os.path.join(data_dir,'offline_diarization_with_asr.yaml')):
    CONFIG = wget.download(CONFIG_URL, data_dir)
else:
    CONFIG = os.path.join(data_dir,'offline_diarization_with_asr.yaml')

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
    with open(os.path.join(data_dir, 'input_manifest.json'), 'w') as fp:
        json.dump(meta, fp)
        fp.write('\n')
