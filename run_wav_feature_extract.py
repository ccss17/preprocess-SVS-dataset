import preprocess_svs as ps

preprocessed_gv_path = "preprocessed_gv/"
preprocessed_mssv_path = "preprocessed_mssv/"
preprocessed_gv_wav_path = "preprocessed_gv/wav/"
preprocessed_mssv_wav_path = "preprocessed_mssv/wav/"

preprocessed_gv_path = "/home/ccss17/dataset/gv_dataset_preprocessed/"
preprocessed_gv_wav_path = "/home/ccss17/dataset/gv_dataset_preprocessed/wav/"
preprocessed_mssv_path = "/home/ccss17/dataset/new_mssv_preprocessed/"
preprocessed_mssv_wav_path = "/home/ccss17/dataset/new_mssv_preprocessed/wav/"

preprocessed_hsd_path = "/home/jpseo/data/hsd/"
preprocessed_hsd_wav_path = "/home/jpseo/data/hsd/wavs/"

# ps.save_all_f0_mel_energy(preprocessed_gv_wav_path, preprocessed_gv_path)
# ps.save_all_f0_mel_energy(preprocessed_mssv_wav_path, preprocessed_mssv_path)
ps.save_all_f0_mel_energy(preprocessed_hsd_wav_path, preprocessed_hsd_path)