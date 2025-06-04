from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import mssv
from preprocess_svs import SVS_Preprocessor

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
