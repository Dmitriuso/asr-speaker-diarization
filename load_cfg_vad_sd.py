import wget
import json

from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

own_file = DATA / 'videos_audios/youtube=p601i1-vTjI.wav'

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization.yaml"

if not (DATA / 'offline_diarization.yaml').exists():
    CONFIG = wget.download(CONFIG_URL, str(DATA))
else:
    CONFIG = DATA / 'offline_diarization_with_asr.yaml'

cfg = OmegaConf.load(CONFIG)


meta = {
    'audio_filepath': own_file,
    'offset': 0,
    'duration':None,
    'label': 'infer',
    'text': '-',
    'num_speakers': None,
    'uem_filepath' : None
}

input_manifest_dir = str(DATA / 'input_manifest_vad_sd.json')
print(type(input_manifest_dir))
with open(input_manifest_dir, 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')