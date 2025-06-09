from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import mssv

mssv_path = "D:/dataset/004.다화자 가창 데이터"
mssv_path = "/home/sjkim/dataset/bertapc/bak/mssv/data"
mssv_midi_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/midi"
mssv_wav_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/wav"
split_json_dirpath = "/home/ccss17/dataset/new_mssv_json_split"

# rprint(ps.find_exclusive_two_type_files("*.mid", "*.wav", mssv_midi_path, mssv_wav_path))
# rprint(mssv.check_abnormal_file(mssv_midi_path))
# rprint(mssv.check_abnormal_file(mssv_wav_path))
# rprint(mssv.rename_abnormal_file(mssv_wav_path))
# rprint(mssv.rename_abnormal_file(mssv_midi_path))
# rprint(mssv.remove_abnormal_file(mssv_midi_path))
# rprint(mssv.remove_abnormal_file(mssv_wav_path))
rprint(
    ps.find_exclusive_two_type_files(
        "*.mid", "*.json", mssv_midi_path, split_json_dirpath
    )
)
