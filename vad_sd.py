import json
import wget

from pathlib import Path
from omegaconf import OmegaConf

from nemo.collections.asr.models import ClusteringDiarizer

from tools.rttm_smoothing import smooth_rttm


ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(parents=True, exist_ok=True)

file_name = 'youtube=p601i1-vTjI'
own_file = str(DATA / f'videos_audios/{file_name}.wav')
bench_rttm = str(DATA / "bench_rttms/Tjl_time_stamps_bench.rttm")

CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization.yaml"

if not (DATA / 'offline_diarization.yaml').exists():
    CONFIG = wget.download(CONFIG_URL, str(DATA))
else:
    CONFIG = DATA / 'offline_diarization_with_asr.yaml'

cfg = OmegaConf.load(CONFIG)

meta = {
    'audio_filepath': own_file,
    'offset': 0,
    'duration': None,
    'label': 'infer',
    'text': '-',
    'num_speakers': None,
    'rttm_filepath': bench_rttm,
    'uem_filepath' : None
}

input_manifest_dir = str(DATA / 'input_manifest_vad_sd.json')
with open(input_manifest_dir, 'w') as fp:
    json.dump(meta, fp)
    fp.write('\n')

pretrained_vad = 'vad_marblenet'
pretrained_speaker_model = 'titanet_large'

cfg.diarizer.manifest_filepath = input_manifest_dir
cfg.diarizer.out_dir = str(OUTPUT) #Directory to store intermediate files and prediction outputs

cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
cfg.diarizer.oracle_vad = False # compute VAD provided with model_path to vad cfg
cfg.diarizer.clustering.parameters.oracle_num_speakers=False

#Here we use our inhouse pretrained NeMo VAD
cfg.diarizer.vad.model_path = pretrained_vad
cfg.diarizer.vad.window_length_in_sec = 0.15
cfg.diarizer.vad.shift_length_in_sec = 0.01
cfg.diarizer.vad.parameters.onset = 0.8
cfg.diarizer.vad.parameters.offset = 0.6
cfg.diarizer.vad.parameters.min_duration_on = 0.1
cfg.diarizer.vad.parameters.min_duration_off = 0.4

sd_model = ClusteringDiarizer(cfg=cfg)

if __name__ == '__main__':
    sd_model.diarize()
    smooth_rttm(str(OUTPUT / f"pred_rttms/{file_name}.rttm"), str(OUTPUT / "pred_rttms"))
