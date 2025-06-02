from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import mssv
from preprocess_svs import SVS_Preprocessor

# -------------------------------------------

mssv_path = "D:/dataset/004.다화자 가창 데이터"

rprint(mssv.find_exclusive_two_type_files("*.mid", "*.wav", mssv_path))
rprint(mssv.check_abnormal_file(mssv_path))
rprint(mssv.rename_abnormal_file(mssv_path))
rprint(mssv.remove_abnormal_file(mssv_path))

# -------------------------------------------

midi_dirpath = "sample/mssv/midi"
json_dirpath = "sample/mssv/json"
split_json_dirpath = "sample/mssv/split_json"
wav_dirpath = "sample/mssv/wav/"
split_json_dirpath = "sample/mssv/split_json/"
preprocessed_mssv_path = "preprocessed_mssv/"

mssv.midis_to_jsons(midi_dirpath, json_dirpath)
mssv.split_jsons(json_dirpath, split_json_dirpath)
mssv.save_duration_pitch_metadata_split_audio(
    wav_dirpath, split_json_dirpath, preprocessed_mssv_path
)

preprocessor = SVS_Preprocessor(
    base_path="preprocessed_mssv",
    model_name="large-v3",
    device="cpu",
    language="ko",
)
preprocessor.process_all_files()


preprocessor.verify_dataset_consistency()

file_path = "preprocessed_mssv/metadata.txt"
ps.g2p_metadata(file_path)
