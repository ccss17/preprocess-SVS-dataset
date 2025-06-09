import preprocess_svs as ps
from preprocess_svs import gv
from preprocess_svs import SVS_Preprocessor

gv_json_path = "/home/sjkim/dataset/bertapc/bak/gv/data/json"
gv_midi_path = "/home/sjkim/dataset/bertapc/bak/gv/data/midi"
gv_wav_path = "/home/sjkim/dataset/bertapc/bak/gv/data/wav_22050"
gv_json_preprocessed_path = "/home/ccss17/dataset/gv_json_preprocessed"
gv_json_split_path = "/home/ccss17/dataset/gv_json_splt"
preprocessed_gv_path = "/home/ccss17/dataset/gv_dataset_preprocessed"
metadata_path = preprocessed_gv_path + "/metadata.txt"
normalized_metadata_path = preprocessed_gv_path + "/normalized_metadata.txt"

print("1: preprocess jsons")
gv.preprocess_json(
    gv_json_path,
    gv_midi_path,
    gv_json_preprocessed_path,
    parallel=True,
)

print("2: split jsons")
ps.split_jsons(gv_json_preprocessed_path, gv_json_split_path)

print("3: save_duration_pitch_metadata_split_audio")
ps.save_duration_pitch_metadata_split_audio(
    gv_wav_path,
    gv_json_split_path,
    preprocessed_gv_path,
    singer_id_from_filepath=gv.singer_id_from_filepath,
)

print("4: normalize lyric")
preprocessor = SVS_Preprocessor(
    base_path=preprocessed_gv_path,
    model_name="large-v3",
    device="cuda",
    language="ko",
)
preprocessor.process_all_files()
preprocessor.verify_dataset_consistency()

print("5: g2p")
ps.g2p_metadata(normalized_metadata_path)
