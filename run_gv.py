from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import gv
from preprocess_svs import SVS_Preprocessor

gv_path = "D:/dataset/177.다음색 가이드보컬 데이터"
gv_json_preprocessed = ""
gv_sample_preprocessed = "sample/gv/json_preprocessed"
split_json_dirpath = ""

# verify file correspondence
jsons = ps.get_files(gv_path, "json", sort=True)
mids = ps.get_files(gv_path, "mid", sort=True)
wavs = ps.get_files(gv_path, "wav", sort=True)
rprint(gv.verify_files_coherent(jsons, mids))
rprint(gv.verify_files_coherent(wavs, mids))
rprint(gv.verify_files_coherent(jsons, wavs))

# remove_abnormal_file
gv.remove_abnormal_file(gv_path)

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
wav_dirpath = "sample/gv/wav/"
split_json_dirpath = "sample/gv/split_json/"
preprocessed_gv_path = ""
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

# Apply g2p
file_path = "preprocessed_gv/metadata.txt"
ps.g2p_metadata(file_path)
