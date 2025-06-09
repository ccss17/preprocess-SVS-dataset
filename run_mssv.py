import re

import preprocess_svs as ps
from preprocess_svs import mssv
from preprocess_svs import SVS_Preprocessor

mssv_midi_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/midi"
mssv_wav_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/wav"
json_dirpath = "/home/ccss17/dataset/new_mssv_json"
split_json_dirpath = "/home/ccss17/dataset/new_mssv_json_split"
preprocessed_mssv_path = "/home/ccss17/dataset/new_mssv_preprocessed"
metadata_path = preprocessed_mssv_path + "/metadata.txt"
pre_metadata_path = preprocessed_mssv_path + "/pre_metadata.txt"

print("1: midis_to_jsons")
mssv.midis_to_jsons(mssv_midi_path, json_dirpath)

print("2: split_jsons")
ps.split_jsons(json_dirpath, split_json_dirpath)

print("3: save_duration_pitch_metadata_split_audio")
ps.save_duration_pitch_metadata_split_audio(
    mssv_wav_path,
    split_json_dirpath,
    preprocessed_mssv_path,
    signer_id_from_filepath=lambda x: int(re.findall(r"s\d\d", x)[0][1:]) + 26,
)

print("4: lyric normalize")
preprocessor = SVS_Preprocessor(
    base_path=preprocessed_mssv_path,
    model_name="large-v3",
    device="cuda",
    language="ko",
)
preprocessor.process_all_files()
preprocessor.verify_dataset_consistency()

print("5: g2p")
ps.g2p_metadata(pre_metadata_path)
