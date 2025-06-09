#from rich import print as rprint

import src.preprocess_svs as ps
from src.preprocess_svs import mssv
from src.preprocess_svs import SVS_Preprocessor

midi_dirpath = "sample/mssv/midi"

mssv_midi_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/midi"
mssv_wav_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/wav"
json_dirpath = "/home/ccss17/dataset/new_mssv_json"
split_json_dirpath = "/home/ccss17/dataset/new_mssv_json_split"
preprocessed_mssv_path = "/home/ccss17/dataset/new_mssv_preprocessed"
metadata_path = preprocessed_mssv_path + "/metadata.txt"
pre_metadata_path = preprocessed_mssv_path + "/pre_metadata.txt"

# print(f"1: midis_to_jsons")
# mssv.midis_to_jsons(mssv_midi_path, json_dirpath)
# print(f"2: split_jsons")
# mssv.split_jsons(json_dirpath, split_json_dirpath)
#print(f"3: save_duration_pitch_metadata_split_audio")
#mssv.save_duration_pitch_metadata_split_audio(
#    mssv_wav_path, split_json_dirpath, preprocessed_mssv_path
#)

print(f"4: lyric normalize")
preprocessor = SVS_Preprocessor(
    base_path=preprocessed_mssv_path,
    model_name="large-v3",
    device="cuda",
    language="ko",
)
preprocessor.process_all_files()
preprocessor.verify_dataset_consistency()
print(f"5: g2p")
ps.g2p_metadata(pre_metadata_path)

