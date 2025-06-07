from rich import print as rprint

import src.preprocess_svs as ps
from src.preprocess_svs import gv
from src.preprocess_svs import SVS_Preprocessor

gv_json_path = "/home/sjkim/dataset/bertapc/bak/gv/data/json"
gv_midi_path = "/home/sjkim/dataset/bertapc/bak/gv/data/midi"
gv_wav_path = "/home/sjkim/dataset/bertapc/bak/gv/data/wav_22050"
gv_json_preprocessed_path = "/home/ccss17/dataset/gv_json_preprocessed"
gv_json_split_path = "/home/ccss17/dataset/gv_json_splt"
preprocessed_gv_path =  "/home/ccss17/dataset/gv_dataset_preprocessed"
metadata_path = preprocessed_gv_path + "/metadata.txt"

# preprocess json
# print(f"1: preprocess jsons")
# gv.preprocess_json(
    # gv_json_path,
    # gv_midi_path,
    # gv_json_preprocessed_path,
    # parallel=True,
# )

# split json
# print(f"2: split jsons")
# gv.split_jsons(gv_json_preprocessed_path, gv_json_split_path)

# save duration, pitch as .npy, split audio, save metadata
#print(f"3: save_duration_pitch_metadata_split_audio")
#gv.save_duration_pitch_metadata_split_audio(
#    gv_wav_path, gv_json_split_path, preprocessed_gv_path
#)
'''
# normalize lyric
print(f"4: normalize lyric")
preprocessor = SVS_Preprocessor(
    base_path=preprocessed_gv_path,
    model_name="large-v3",
    device="cuda",
    language="ko",
)
preprocessor.process_all_files()
preprocessor.verify_dataset_consistency()

# Apply g2p
print(f"5: g2p")
ps.g2p_metadata(metadata_path)
'''
