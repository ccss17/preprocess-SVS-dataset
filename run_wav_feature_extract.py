import preprocess_svs as ps

preprocessed_gv_path = "preprocessed_gv/"
preprocessed_mssv_path = "preprocessed_mssv/"
preprocessed_gv_wav_path = "preprocessed_gv/wav/"
preprocessed_mssv_wav_path = "preprocessed_mssv/wav/"

ps.save_all_f0_mel_energy(preprocessed_gv_wav_path, preprocessed_gv_path)
ps.save_all_f0_mel_energy(preprocessed_mssv_wav_path, preprocessed_mssv_path)
