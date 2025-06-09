from pathlib import Path
import re
import itertools
import pathlib
import multiprocessing as mp
import sys

import pandas as pd
import midii

from .util import (
    get_files,
    _preprocess_sort_by_start_time,
    _preprocess_add_frame_col,
    _preprocess_remove_front_back_silence,
    _preprocess_silence_pitch_zero,
    _preprocess_merge_silence,
    _preprocess_remove_short_silence,
)


def midi_to_note_list(midi_filepath, quantize=True):
    try:
        mid = midii.MidiFile(
            midi_filepath, convert_1_to_0=True, lyric_encoding="utf-8"
        )
        mid.lyrics
    except:  # noqa: E722
        mid = midii.MidiFile(
            midi_filepath, convert_1_to_0=True, lyric_encoding="cp949"
        )
    if quantize:
        mid.quantize(unit="32")
    data = []
    total_duration = 0
    residual_duration = 0
    active_note = {}
    silence_note = {}
    for msg in mid:
        msg_end_time = total_duration + msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            residual_duration += msg.time
            if residual_duration > 0:
                if not silence_note:
                    silence_note = {
                        "start_time": total_duration,
                        "pitch": 0,
                        "lyric": " ",
                    }
                silence_note["end_time"] = msg_end_time
                silence_note["duration"] = (
                    msg_end_time - silence_note["start_time"]
                )
                data.append(silence_note.copy())
                silence_note.clear()
                residual_duration = 0
            active_note = {
                "start_time": msg_end_time,
                "pitch": msg.note,
            }
        elif msg.type == "lyrics":
            active_note["lyric"] = midii.MessageAnalyzer_lyrics(
                msg=msg, encoding=mid.lyric_encoding
            ).lyric
        elif msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            active_note["end_time"] = msg_end_time
            active_note["duration"] = msg_end_time - active_note["start_time"]
            data.append(active_note.copy())
            active_note.clear()
        else:
            if not active_note and not silence_note:
                silence_note = {
                    "start_time": total_duration,
                    "pitch": 0,
                    "lyric": " ",
                }
            if not active_note:
                residual_duration += msg.time
        total_duration = msg_end_time

    return data


def _preprocess_slice_actual_lyric(df):
    j_indices = df.index[df["lyric"] == "J"].tolist()
    idx_j = j_indices[0]
    h_indices = df.index[df["lyric"] == "H"].tolist()
    idx_h = h_indices[0]
    slice_start_index = idx_j + 1
    slice_end_index = idx_h
    df = df.iloc[slice_start_index:slice_end_index].reset_index(drop=True)
    return df


def preprocess_notes(notes, out_json_path):
    df = pd.DataFrame(notes)

    # ["J":"H"]
    df = _preprocess_slice_actual_lyric(df)
    # sort by time
    df = _preprocess_sort_by_start_time(df)
    # remove front & back silence
    df = _preprocess_remove_front_back_silence(df)
    # lyric=" " --> pitch=0
    df = _preprocess_silence_pitch_zero(df)
    # merge lyric=" " items
    df = _preprocess_merge_silence(df)
    # remove silence < 0.3
    df = _preprocess_remove_short_silence(df, 0.3)
    # add frames col
    df = _preprocess_add_frame_col(df)
    # to json
    json_filepath = Path(out_json_path)
    json_filepath.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(
        json_filepath,
        orient="records",
        indent=4,
        force_ascii=False,
    )


def check_abnormal_file(mssv_dir):
    """
    mssv 데이터셋 사전 준비:
    - 삭제
        - 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_06799_+0_a_s01_f_02.mid(이유: 대응되는 wav 가 없음)
        - 004.다화자 가창 데이터/01.데이터/1.Training/원천데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_06975_+0_a_s01_f_02.wav(이유: 대응되는 mid 가 없음)
        - 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_00118_+0_a_s01_f_02.mid 와 그에 대응되는 wav 파일(이유: MIDI 파일 내부 데이터가 깨져있음)
        - 다음의 mid 파일과 대응되는 wav 파일(이유: MIDI note_on - lyrics - note_off 패턴 불일치)
            - 00: (마지막 "H" 라벨 하나만 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s03/ro_03036_+0_a_s03_f_02.mid
            - 01: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02207_-3_a_s13_m_04.mid
            - 02: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02314_-2_a_s13_m_04.mid
            - 03: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07373_+0_a_s17_f_04.mid
            - 04: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07556_+4_a_s17_f_04.mid
            - 05: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_10416_+0_a_s17_f_04.mid
            - 06: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_14574_-7_s_s15_m_04.mid
            - 07: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_19085_+0_s_s15_m_04.mid
            - 08: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_09190_+0_s_s16_f_04.mid
            - 09: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_10836_+5_s_s16_f_04.mid
            - 10: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/A. 남성/03. 40대 이상/가창자_s18/ro_07463_+0_s_s18_m_04.mid
            - 11: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_04993_+0_a_s14_f_03.mid
            - 12: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10604_+5_a_s14_f_03.mid
            - 13: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10901_+3_a_s14_f_03.mid
            - 14: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_11017_+5_a_s14_f_03.mid
            - 15: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_04602_+0_a_s13_m_04.mid
            - 16: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_05683_-3_a_s13_m_04.mid
            - 17: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_07020_+0_a_s13_m_04.mid
            - 18: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_11088_-7_a_s13_m_04.mid
        - 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s03/ro_03036_+0_a_s03_f_02.mid 와 대응되는 wav 파일(이유: lyrics 이 time 값을 가짐)
    """
    abnormal_files = (
        "ba_06799_+0_a_s01_f_02.mid",
        "ba_06975_+0_a_s01_f_02.wav",
        "ba_00118_+0_a_s01_f_02.mid",
        "ba_00118_+0_a_s01_f_02.wav",
        "ro_03036_+0_a_s03_f_02.mid",
        "ba_02207_-3_a_s13_m_04.mid",
        "ba_02314_-2_a_s13_m_04.mid",
        "tr_07373_+0_a_s17_f_04.mid",
        "r_07556_+4_a_s17_f_04.mid",
        "tr_10416_+0_a_s17_f_04.mid",
        "tr_14574_-7_s_s15_m_04.mid",
        "tr_19085_+0_s_s15_m_04.mid",
        "ro_09190_+0_s_s16_f_04.mid",
        "ro_10836_+5_s_s16_f_04.mid",
        "ro_07463_+0_s_s18_m_04.mid",
        "ba_04993_+0_a_s14_f_03.mid",
        "ba_10604_+5_a_s14_f_03.mid",
        "ba_10901_+3_a_s14_f_03.mid",
        "ba_11017_+5_a_s14_f_03.mid",
        "ba_04602_+0_a_s13_m_04.mid",
        "ba_05683_-3_a_s13_m_04.mid",
        "ba_07020_+0_a_s13_m_04.mid",
        "ba_11088_-7_a_s13_m_04.mid",
        "ro_03036_+0_a_s03_f_02.wav",
        "ba_02207_-3_a_s13_m_04.wav",
        "ba_02314_-2_a_s13_m_04.wav",
        "tr_07373_+0_a_s17_f_04.wav",
        "r_07556_+4_a_s17_f_04.wav",
        "tr_10416_+0_a_s17_f_04.wav",
        "tr_14574_-7_s_s15_m_04.wav",
        "tr_19085_+0_s_s15_m_04.wav",
        "ro_09190_+0_s_s16_f_04.wav",
        "ro_10836_+5_s_s16_f_04.wav",
        "ro_07463_+0_s_s18_m_04.wav",
        "ba_04993_+0_a_s14_f_03.wav",
        "ba_10604_+5_a_s14_f_03.wav",
        "ba_10901_+3_a_s14_f_03.wav",
        "ba_11017_+5_a_s14_f_03.wav",
        "ba_04602_+0_a_s13_m_04.wav",
        "ba_05683_-3_a_s13_m_04.wav",
        "ba_07020_+0_a_s13_m_04.wav",
        "ba_11088_-7_a_s13_m_04.wav",
    )

    mssv_dir = pathlib.Path(mssv_dir)
    existing_files_set = {p.name for p in mssv_dir.rglob("*") if p.is_file()}
    abnormal_files_set = set(abnormal_files)
    # missing_files = list(abnormal_files_set.difference(existing_files_set))
    found_files = list(abnormal_files_set.intersection(existing_files_set))
    # return missing_files, found_files
    return found_files


def remove_abnormal_file(mssv_dir):
    """
    mssv 데이터셋 사전 준비:
    - 삭제
        - 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_06799_+0_a_s01_f_02.mid(이유: 대응되는 wav 가 없음)
        - 004.다화자 가창 데이터/01.데이터/1.Training/원천데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_06975_+0_a_s01_f_02.wav(이유: 대응되는 mid 가 없음)
        - 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/B. 여성/01. 20대/가창자_s01/ba_00118_+0_a_s01_f_02.mid 와 그에 대응되는 wav 파일(이유: MIDI 파일 내부 데이터가 깨져있음)
        - 다음의 mid 파일과 대응되는 wav 파일(이유: MIDI note_on - lyrics - note_off 패턴 불일치)
            - 00: (마지막 "H" 라벨 하나만 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s03/ro_03036_+0_a_s03_f_02.mid
            - 01: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02207_-3_a_s13_m_04.mid
            - 02: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_02314_-2_a_s13_m_04.mid
            - 03: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07373_+0_a_s17_f_04.mid
            - 04: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_07556_+4_a_s17_f_04.mid
            - 05: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/B. 여성/03. 40대 이상/가창자_s17/tr_10416_+0_a_s17_f_04.mid
            - 06: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_14574_-7_s_s15_m_04.mid
            - 07: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/03.트로트/A. 남성/03. 40대 이상/가창자_s15/tr_19085_+0_s_s15_m_04.mid
            - 08: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_09190_+0_s_s16_f_04.mid
            - 09: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/03. 40대 이상/가창자_s16/ro_10836_+5_s_s16_f_04.mid
            - 10: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/A. 남성/03. 40대 이상/가창자_s18/ro_07463_+0_s_s18_m_04.mid
            - 11: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_04993_+0_a_s14_f_03.mid
            - 12: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10604_+5_a_s14_f_03.mid
            - 13: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_10901_+3_a_s14_f_03.mid
            - 14: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/02. 30대/가창자_s14/ba_11017_+5_a_s14_f_03.mid
            - 15: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_04602_+0_a_s13_m_04.mid
            - 16: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_05683_-3_a_s13_m_04.mid
            - 17: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_07020_+0_a_s13_m_04.mid
            - 18: (전체 note 패턴이 깨져있음) 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/A. 남성/03. 40대 이상/가창자_s13/ba_11088_-7_a_s13_m_04.mid
        - 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s03/ro_03036_+0_a_s03_f_02.mid 와 대응되는 wav 파일(이유: lyrics 이 time 값을 가짐)
    """
    abnormal_files = (
        "ba_06799_+0_a_s01_f_02.mid",
        "ba_06975_+0_a_s01_f_02.wav",
        "ba_00118_+0_a_s01_f_02.mid",
        "ba_00118_+0_a_s01_f_02.wav",
        "ro_03036_+0_a_s03_f_02.mid",
        "ba_02207_-3_a_s13_m_04.mid",
        "ba_02314_-2_a_s13_m_04.mid",
        "tr_07373_+0_a_s17_f_04.mid",
        "r_07556_+4_a_s17_f_04.mid",
        "tr_10416_+0_a_s17_f_04.mid",
        "tr_14574_-7_s_s15_m_04.mid",
        "tr_19085_+0_s_s15_m_04.mid",
        "ro_09190_+0_s_s16_f_04.mid",
        "ro_10836_+5_s_s16_f_04.mid",
        "ro_07463_+0_s_s18_m_04.mid",
        "ba_04993_+0_a_s14_f_03.mid",
        "ba_10604_+5_a_s14_f_03.mid",
        "ba_10901_+3_a_s14_f_03.mid",
        "ba_11017_+5_a_s14_f_03.mid",
        "ba_04602_+0_a_s13_m_04.mid",
        "ba_05683_-3_a_s13_m_04.mid",
        "ba_07020_+0_a_s13_m_04.mid",
        "ba_11088_-7_a_s13_m_04.mid",
        "ro_03036_+0_a_s03_f_02.wav",
        "ba_02207_-3_a_s13_m_04.wav",
        "ba_02314_-2_a_s13_m_04.wav",
        "tr_07373_+0_a_s17_f_04.wav",
        "r_07556_+4_a_s17_f_04.wav",
        "tr_10416_+0_a_s17_f_04.wav",
        "tr_14574_-7_s_s15_m_04.wav",
        "tr_19085_+0_s_s15_m_04.wav",
        "ro_09190_+0_s_s16_f_04.wav",
        "ro_10836_+5_s_s16_f_04.wav",
        "ro_07463_+0_s_s18_m_04.wav",
        "ba_04993_+0_a_s14_f_03.wav",
        "ba_10604_+5_a_s14_f_03.wav",
        "ba_10901_+3_a_s14_f_03.wav",
        "ba_11017_+5_a_s14_f_03.wav",
        "ba_04602_+0_a_s13_m_04.wav",
        "ba_05683_-3_a_s13_m_04.wav",
        "ba_07020_+0_a_s13_m_04.wav",
        "ba_11088_-7_a_s13_m_04.wav",
        "tr_07556_+4_a_s17_f_04.mid",
    )
    deleted_files = []
    error_files = []
    target_filenames_set = set(abnormal_files)
    mssv_dir = pathlib.Path(mssv_dir)
    for path_object in mssv_dir.rglob("*"):
        if path_object.is_file() and path_object.name in target_filenames_set:
            try:
                path_object.unlink()
                deleted_files.append(str(path_object))
            except OSError as e:
                error_files.append(str(path_object))
                print(
                    f"Removing Error: '{path_object}' - {e}",
                    file=sys.stderr,
                )

    return deleted_files, error_files


def rename_abnormal_file(mssv_dir):
    """
    mssv 데이터셋 사전 준비:
    - 경로 변경
        - 004.다화자 가창 데이터/01.데이터/2.Validation/원천데이터/02.록팝/A. 남성/01. 20대/가창자_s02/ro_01274_+0_m_s02_m_02.wav 을 "ro_01274_+0_a_s02_m_02.wav" 로 변경(이유: 004.다화자 가창 데이터/01.데이터/2.Validation/라벨링데이터/02.록팝/A. 남성/01. 20대/가창자_s02/ro_01274_+0_a_s02_m_02.mid 와 경로가 대응되도록)
        - 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/03. 40대 이상/가창자_s16/ba_10754_+2_s_yej_f_04.mid, 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/01.발라드R&B/B. 여성/03. 40대 이상/가창자_s16/ba_10754_+2_s_yej_f_04.json, 004.다화자 가창 데이터/01.데이터/1.Training/원천데이터/01.발라드R&B/B. 여성/03. 40대 이상/가창자_s16/ba_10754_+2_s_yej_f_04.wav 의 경로에서 "yej" 를 "s16" 로 변경(이유: 같은 디렉토리 내 경로 일관성 유지)
        - 004.다화자 가창 데이터/01.데이터/1.Training/라벨링데이터/02.록팝/B. 여성/01. 20대/가창자_s06/ro_23930_-2_a_lsb_f_02.mid, 004.다화자 가창 데이터/01.데이터/1.Training/원천데이터/02.록팝/B. 여성/01. 20대/가창자_s06/ro_23930_-2_a_lsb_f_02.wav 의 경로에서 "lsb" --> "s06" 로 변경(이유: 같은 디렉토리 내 경로 일관성 유지)
    """
    rename_mapping = {
        "ro_01274_+0_m_s02_m_02.wav": "ro_01274_+0_a_s02_m_02.wav",
        "ba_10754_+2_s_yej_f_04.mid": "ba_10754_+2_s_s16_f_04.mid",
        "ba_10754_+2_s_yej_f_04.json": "ba_10754_+2_s_s16_f_04.json",
        "ba_10754_+2_s_yej_f_04.wav": "ba_10754_+2_s_s16_f_04.wav",
        "ro_23930_-2_a_lsb_f_02.mid": "ro_23930_-2_a_s06_f_02.mid",
        "ro_23930_-2_a_lsb_f_02.wav": "ro_23930_-2_a_s06_f_02.wav",
    }
    mssv_dir = pathlib.Path(mssv_dir)
    renamed_log = {}
    error_log = {}

    for current_path in mssv_dir.rglob("*"):
        if current_path.is_file():
            current_name = current_path.name
            if current_name in rename_mapping:
                new_name = rename_mapping[current_name]
                new_path = current_path.with_name(new_name)
                try:
                    current_path.rename(new_path)
                    renamed_log[str(current_path)] = str(new_path)
                except OSError as e:
                    error_log[str(current_path)] = str(e)
                    print(f"Rename Error: '{current_path}' - {e}")

    return renamed_log, error_log


def verify_midi_pattern_on_lyrics_off(midi_path):
    mid = midii.MidiFile(midi_path, convert_1_to_0=True)
    note_info_msg_set = ("note_on", "lyrics", "note_off")
    infinite_cycler = itertools.cycle(note_info_msg_set)
    for msg in mid:
        if msg.type in note_info_msg_set:
            if not msg.type == next(infinite_cycler):
                return False
    return True


def verify_midi_lyrics_has_no_time(midi_path):
    mid = midii.MidiFile(midi_path, convert_1_to_0=True)
    for msg in mid:
        if msg.type == "lyrics":
            if msg.time != 0:
                return False
    return True


def verify_midi_pattern_on_lyrics_off_return_path(midi_path):
    return "" if verify_midi_pattern_on_lyrics_off(midi_path) else midi_path


def verify_midi_lyrics_has_no_time_return_path(midi_path):
    return "" if verify_midi_lyrics_has_no_time(midi_path) else midi_path


def verify_midi_files_pattern_on_lyrics_off(dir_path, parallel=False):
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            i = 0
            for pattern_broken_midi_path in p.map(
                verify_midi_pattern_on_lyrics_off_return_path,
                get_files(dir_path, "mid"),
            ):
                if pattern_broken_midi_path:
                    print(i, pattern_broken_midi_path)
                    i += 1
    else:
        i = 0
        for midi_file in get_files(dir_path, "mid"):
            if not verify_midi_pattern_on_lyrics_off(midi_file):
                print(i, midi_file)
                i += 1


def verify_midi_files_lyrics_has_no_time(dir_path, parallel=False):
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            i = 0
            for abnormal_midi_path in p.map(
                verify_midi_lyrics_has_no_time_return_path,
                get_files(dir_path, "mid"),
            ):
                if abnormal_midi_path:
                    print(i, abnormal_midi_path)
                    i += 1
    else:
        i = 0
        for midi_file in get_files(dir_path, "mid"):
            if verify_midi_lyrics_has_no_time_return_path(midi_file):
                print(i, midi_file)
                i += 1


def midi_to_json(midi_filepath, json_filepath):
    notes = midi_to_note_list(midi_filepath)
    try:
        preprocess_notes(notes, json_filepath)
    except:
        return


def midis_to_jsons(midi_dirpath, json_dirpath):
    json_dirpath = Path(json_dirpath)
    with mp.Pool(mp.cpu_count()) as p:
        args = []
        mid_paths = get_files(midi_dirpath, "mid", sort=True)
        for mid_path in mid_paths:
            arg = (
                mid_path,
                json_dirpath / Path(mid_path).with_suffix(".json").name,
            )
            args.append(arg)
        p.starmap(midi_to_json, args)


def singer_id_from_filepath(filepath):
    return int(re.findall(r"s\d\d", filepath)[0][1:]) + 26
