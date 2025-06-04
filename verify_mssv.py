from rich import print as rprint

from preprocess_svs import mssv

mssv_path = "D:/dataset/004.다화자 가창 데이터"
mssv_path = "/home/sjkim/dataset/bertapc/bak/mssv/data"
mssv_midi_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/midi"
mssv_wav_path = "/home/sjkim/dataset/bertapc/bak/mssv/data/wav"

rprint(mssv.find_exclusive_two_type_files("*.mid", "*.wav", mssv_path))
# rprint(mssv.check_abnormal_file(mssv_path))
# rprint(mssv.rename_abnormal_file(mssv_path))
# rprint(mssv.remove_abnormal_file(mssv_path))

