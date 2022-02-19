import wget
import gzip
import shutil
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE

import pprint
# pp = pprint.PrettyPrinter(indent=4)

from load_config import data_dir, cfg


def gunzip(file_path, output_path):
    with gzip.open(file_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
        f_in.close()
        f_out.close()


lm_file = data_dir / '4gram_big.arpa'

if lm_file.exists():
    pass
else:
    ARPA_URL = 'https://kaldi-asr.org/models/5/4gram_big.arpa.gz'
    f = wget.download(ARPA_URL, data_dir)
    gunzip(f, f.replace(".gz", ""))

cfg.diarizer.manifest_filepath = str(data_dir / 'input_manifest.json')

pretrained_speaker_model = 'ecapa_tdnn'
cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
cfg.diarizer.out_dir = str(data_dir) #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
cfg.diarizer.clustering.parameters.oracle_num_speakers = False

# Using VAD generated from ASR timestamps
asr_ckpt = 'stt_fr_citrinet_1024_gamma_0_25' #'stt_fr_quartznet15x5.nemo'
cfg.diarizer.asr.model_path = asr_ckpt # 'stt_fr_quartznet15x5'
cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD
cfg.diarizer.asr.parameters.asr_batch_size = 2
cfg.diarizer.asr.parameters.asr_based_vad = True
cfg.diarizer.asr.parameters.threshold = 20 # ASR based VAD threshold: If 100, all silences under 1 sec are ignored.
cfg.diarizer.asr.parameters.decoder_delay_in_sec = None # Decoder delay is compensated for 0.2 sec
cfg.diarizer.asr.parameters.lenient_overlap_WDER = True
cfg.diarizer.asr.parameters.colored_text = False
cfg.diarizer.asr.parameters.break_lines = False

arpa_model_path = str(data_dir / '4gram_big.arpa')
# cfg.diarizer.asr.ctc_decoder_parameters.pretrained_language_model = arpa_model_path

cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model = arpa_model_path
cfg.diarizer.asr.realigning_lm_parameters.logprob_diff_threshold = 1.2


asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
asr_model = asr_ts_decoder.set_asr_model()
word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)

asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer)
asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset

diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
predicted_speaker_label_rttm_path = str(data_dir / 'pred_rttms/mondialisation_full.rttm')

asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
