from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import gv
from preprocess_svs import SVS_Preprocessor


gv_path = "D:/dataset/177.다음색 가이드보컬 데이터"

gv_json_preprocessed = ""
split_json_dirpath = ""
wav_dirpath = "sample/gv/wav/"
split_json_dirpath = "sample/gv/split_json/"
preprocessed_gv_path = ""
metadata_path = "preprocessed_gv/metadata.txt"

# preprocess json
gv.preprocess_json(
    gv_path,
    gv_path,
    gv_json_preprocessed,
    parallel=True,
)

# split json
gv.split_jsons(gv_json_preprocessed, split_json_dirpath)

# save duration, pitch as .npy, split audio, save metadata
gv.save_duration_pitch_metadata_split_audio(
    wav_dirpath, split_json_dirpath, preprocessed_gv_path
)

# normalize lyric
preprocessor = SVS_Preprocessor(
    base_path="preprocessed_gv",
    model_name="large-v3",
    device="cpu",
    language="ko",
)
preprocessor.process_all_files()

preprocessor.verify_dataset_consistency()

# Apply g2p
ps.g2p_metadata(metadata_path)
