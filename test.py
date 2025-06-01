import json
import os

from rich import print as rprint
import midii

import preprocess_svs as ps


def view_gv_forever(dir_path):
    for mid_path in ps.get_files(dir_path, "mid"):
        os.system("clear")
        print(mid_path)
        mid = midii.MidiFile(mid_path, lyric_encoding="cp949")
        mid.print_tracks()
        input()


def test_preprocess_mssv_file():
    midi_filepath = "sample/mssv/midi/ba_05688_-4_a_s02_m_02.mid"
    wav_filepath = "sample/mssv/wav/ba_05688_-4_a_s02_m_02.wav"
    json_filepath = "sample/mssv/json/ba_05688_-4_a_s02_m_02.json"
    split_json_filepath = "sample/mssv/split_json/ba_05688_-4_a_s02_m_02.json"
    preprocessed_mssv_path = "preprocessed_mssv/"
    preprocessed_mssv_duration_path = "preprocessed_mssv/duration"
    preprocessed_mssv_pitch_path = "preprocessed_mssv/pitch"
    preprocessed_mssv_wav_path = "preprocessed_mssv/wav"

    #
    # Step 1 - note duration quantization, save as json
    #
    df_notes = ps.mssv_midi_to_dataframe(midi_filepath)
    df_notes.to_json(
        json_filepath,
        orient="records",
        indent=4,
        force_ascii=False,
    )
    #
    # Step 2 - split notes by silence
    #
    with open(split_json_filepath, "w", encoding="utf-8") as f:
        json.dump(
            ps.split_json_by_silence(json_filepath, min_length=6),
            f,
            indent=4,
            ensure_ascii=False,
        )
    #
    # Step 3 or Step 4 - regularization korean (metadata.txt 의 가사의 글자 갯수가 split 된 duration/pitch/wav 의 갯수와 일치해야 하는지? 만약 일치하지 않아도 된다면, step 4 에 해도 되고, json 이 아니라 kor seq/pitch seq/GT 만 받아도 해도 된다)
    #
    split_json_filepath
    #
    # Step 4 - save duration, pitch as npy file, split audio, save metadata
    #
    metadata_list = []
    metadata_list.append(
        ps.preprocess_mssv_one(
            wav_filepath,
            split_json_filepath,
            preprocessed_mssv_pitch_path,
            preprocessed_mssv_duration_path,
            preprocessed_mssv_wav_path,
        )
    )
    with open(
        f"{preprocessed_mssv_path}/metadata.txt", "w", encoding="utf-8"
    ) as f:
        f.write("".join(metadata_list))


def insect_broken_pattern_mssv_midi_files(mssv_dir):
    t = (
        "004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s03/ro_03036_+0_a_s03_f_02.mid",
        "004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02207_-3_a_s13_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02314_-2_a_s13_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07373_+0_a_s17_f_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07556_+4_a_s17_f_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_10416_+0_a_s17_f_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_14574_-7_s_s15_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_19085_+0_s_s15_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_09190_+0_s_s16_f_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_10836_+5_s_s16_f_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/A. 남성/03. 40대 이상/가창자_s18/ro_07463_+0_s_s18_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_04993_+0_a_s14_f_03.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10604_+5_a_s14_f_03.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10901_+3_a_s14_f_03.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_11017_+5_a_s14_f_03.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_04602_+0_a_s13_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_05683_-3_a_s13_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_07020_+0_a_s13_m_04.mid",
        "004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_11088_-7_a_s13_m_04.mid",
    )
    for f in t:
        midii.MidiFile(mssv_dir + f, convert_1_to_0=True).print_tracks()


def test_tempo_rank():
    sample_mssv_midi = "sample/mssv/midi/ba_05688_-4_a_s02_m_02.mid"
    mid = midii.MidiFile(sample_mssv_midi, convert_1_to_0=True)
    tempo_rank = mid.tempo_rank()
    rprint(tempo_rank)
    rprint(ps.calculate_top_tempo_percentage(tempo_rank))


if __name__ == "__main__":
    # -------------------- Testing --------------------
    test_tempo_rank()

    # -------------------- MSSV --------------------
    sample_mssv_midi = "sample/mssv/midi/ba_05688_-4_a_s02_m_02.mid"
    sample_mssv = "sample/mssv"
    mssv = "D:/dataset/004.다화자 가창 데이터"
    # insect_broken_pattern_mssv_midi_files(mssv_dir="D:/dataset/")
    # ps.verify_midi_files_pattern_on_lyrics_off(mssv, parallel=True)
    # ps.verify_midi_files_lyrics_has_no_time(mssv, parallel=True)
    # rprint(ps.remove_abnormal_mssv_file(mssv))
    # rprint(ps.rename_abnormal_mssv_file(mssv))
    # rprint(ps.find_exclusive_two_type_files("*.mid", "*.wav", mssv))
    # rprint(ps.check_abnormal_mssv_file(mssv))

    # test_preprocess_mssv_file()
    # mid = midii.MidiFile(sample_mssv_midi, convert_1_to_0=True)
    # tempo_rank = mid.tempo_rank()
    # rprint(tempo_rank)
    # rprint(ps.calculate_top_tempo_percentage(tempo_rank))

    # test_tempo_rank(mssv, parallel=True)

    # 이전 end_time == 현재 start_time 검증

    # -------------------- GV --------------------
    gv = "D:/dataset/177.다음색 가이드보컬 데이터"
    gv_json_sample = "sample/gv/json"
    gv_mid_sample = "sample/gv/midi"
    gv_sample_preprocessed = "sample/gv/json_preprocessed"
    gv_json_time_adjusted = "D:/dataset/다음색 가이드보컬 데이터 time_adjusted"
    gv_json_preprocessed = (
        "D:/dataset/다음색 가이드보컬 데이터 json preprocessed"
    )
    midi_filepath = (
        "sample/gv/midi/SINGER_16_10TO29_CLEAR_FEMALE_BALLAD_C0632.mid"
    )
    time_adjusted_json_filepath = "sample/gv/json_time_adjusted/SINGER_16_10TO29_CLEAR_FEMALE_BALLAD_C0632.json"
    filled_time_gaps_json_filepath = "sample/gv/json_filled_time_gaps/SINGER_16_10TO29_CLEAR_FEMALE_BALLAD_C0632.json"

    # print(len(list(get_files(gv, 'mid'))))

    # view_gv_forever(gv)
    # -> midi 가 엉망이므로 json 을 다뤄야 함

    # mid = midii.MidiFile(midi_filepath, convert_1_to_0=True)
    # tempo_rank = mid.tempo_rank()
    # rprint(tempo_rank)
    # rprint(calculate_top_tempo_percentage(tempo_rank))
    # test_tempo_rank(gv, parallel=True, verbose=True)
    # -> json 을 처리하려면 quantize 를 위한 tempo 가 필요한데 json 에는 tempo
    # 정보가 없음 -> tempo rank 검사 -> tempo 가 변하지 않는다는 충분한 보장
    # -> dominate tempo 를 채택하여 quantize 해도 된다

    # verify_json_notes_sorted_by_time(gv, parallel=True)
    # -> 이전 end_time 이 현재 start_time 보다 큰 경우가 있음 -> 이전 end_time 에
    # 현재 start_time 을 맞추면, 뒤따라오는 메시지들의 sync 가 다 틀어짐 -> 이전 end_time
    # 을 현재 start_time 에 맞춰주는 게 더 나음

    # adjust_note_times_sample()
    # adjust_note_times_gv(
    #     gv, "D:/dataset/다음색 가이드보컬 데이터 time_adjusted/"
    # )

    # 노트 사이에 공백음 채워넣기
    # fill_time_gaps_save(
    #     time_adjusted_json_filepath, filled_time_gaps_json_filepath
    # )

    # json vs wav vs mid 대응 되는지
    # remove_abnormal_gv_file(gv)
    # jsons = get_files(gv, "json", sort=True)
    # mids = get_files(gv, "mid", sort=True)
    # wavs = get_files(gv, "wav", sort=True)
    # rprint(verify_files_coherent(jsons, mids))
    # rprint(verify_files_coherent(wavs, mids))
    # rprint(verify_files_coherent(jsons, wavs))

    # gv json -> adjust note times + fill time gaps + quantization + frames
    # preprocess_gv_json(
    #     gv_json_sample, gv_mid_sample, gv_sample_preprocessed, parallel=True
    # )
    # preprocess_gv_json(
    #     gv,
    #     gv,
    #     gv_json_preprocessed,
    #     parallel=True,
    # )
