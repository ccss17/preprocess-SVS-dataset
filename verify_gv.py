from rich import print as rprint

import preprocess_svs as ps
from preprocess_svs import gv


gv_path = "D:/dataset/177.다음색 가이드보컬 데이터"

# verify file correspondence
jsons = ps.get_files(gv_path, "json", sort=True)
mids = ps.get_files(gv_path, "mid", sort=True)
wavs = ps.get_files(gv_path, "wav", sort=True)
rprint(gv.verify_files_coherent(jsons, mids))
rprint(gv.verify_files_coherent(wavs, mids))
rprint(gv.verify_files_coherent(jsons, wavs))

# remove_abnormal_file
rprint(gv.remove_abnormal_file(gv_path))
