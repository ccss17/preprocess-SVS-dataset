from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import gv


gv_json_path = "/home/sjkim/dataset/bertapc/bak/gv/data/json"
gv_midi_path = "/home/sjkim/dataset/bertapc/bak/gv/data/midi"
gv_wav_path = "/home/sjkim/dataset/bertapc/bak/gv/data/wav_22050"

# verify file correspondence
jsons = ps.get_files(gv_json_path, "json", sort=True)
mids = ps.get_files(gv_midi_path, "mid", sort=True)
wavs = ps.get_files(gv_wav_path, "wav", sort=True)
rprint(gv.verify_files_coherent(jsons, mids))
rprint(gv.verify_files_coherent(wavs, mids))
rprint(gv.verify_files_coherent(jsons, wavs))

# remove_abnormal_file
# rprint(gv.remove_abnormal_file(gv_path))
