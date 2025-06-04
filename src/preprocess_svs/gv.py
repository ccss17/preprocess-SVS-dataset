import re
import json
from pathlib import Path
import multiprocessing as mp
import sys
import itertools

import mido
import pandas as pd
import midii
import numpy as np
import librosa
import soundfile
from g2pk2 import G2p


from .util import (
    get_files,
    print_stat,
    _preprocess_sort_by_start_time,
    _preprocess_add_frame_col,
    _preprocess_remove_front_back_silence,
    _preprocess_silence_pitch_zero,
    _preprocess_merge_silence,
    _preprocess_remove_short_silence,
    split_json_by_silence,
)


def adjust_note_times(notes):
    processed_notes = []
    for i, note in enumerate(notes):
        note["start_time"] = float(note["start_time"])
        note["end_time"] = float(note["end_time"])
        if i > 0 and notes[i - 1]["end_time"] > notes[i]["start_time"]:
            notes[i - 1]["end_time"] = notes[i]["start_time"]
            duration = notes[i - 1]["end_time"] - notes[i - 1]["start_time"]
            if duration < 0:
                duration = 0.0
            notes[i - 1]["length"] = duration
        else:
            duration = float(note["length"])
        processed_notes.append(
            {
                "start_time": note["start_time"],
                "end_time": note["end_time"],
                "pitch": note["midi_num"],
                "lyric": note["lyric"],
                "duration": duration,
            }
        )
    return processed_notes


def fill_time_gaps_with_silence(notes):
    filled_notes = []
    last_end_time = 0.0
    first_note_start_time = notes[0]["start_time"]
    if first_note_start_time > 0:
        silence_duration = first_note_start_time - 0.0
        filled_notes.append(
            {
                "start_time": 0.0,
                "end_time": first_note_start_time,
                "pitch": 0,
                "lyric": "",
                "duration": silence_duration,
            }
        )
        last_end_time = first_note_start_time
    for i, note in enumerate(notes):
        if note["start_time"] > last_end_time:
            filled_notes.append(
                {
                    "start_time": last_end_time,
                    "end_time": note["start_time"],
                    "pitch": 0,
                    "lyric": "",
                    "duration": note["start_time"] - last_end_time,
                }
            )
        filled_notes.append(note)
        last_end_time = note["end_time"]
    return filled_notes


def _preprocess_json(json_path, mid_path, out_path):
    out_path = Path(out_path)
    json_path = Path(json_path)
    mid_path = Path(mid_path)
    if not json_path.stem == mid_path.stem == out_path.stem:
        raise ValueError
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data.get("notes")

    # adjust note times
    notes = adjust_note_times(notes)
    # fill time gaps_with silence
    notes = fill_time_gaps_with_silence(notes)

    # lyric "" --> " "
    df = pd.DataFrame(notes)
    df.loc[df["lyric"] == "", "lyric"] = " "

    # quantize
    mid = midii.MidiFile(mid_path, convert_1_to_0=True)
    unit_beats = midii.NOTE["n/32"].beat
    unit_ticks = midii.beat2tick(unit_beats, ticks_per_beat=mid.ticks_per_beat)
    top_tempo = mid.tempo_rank()[0][0]
    unit_seconds = mido.tick2second(
        unit_ticks, ticks_per_beat=mid.ticks_per_beat, tempo=top_tempo
    )
    quantized_durations, err = midii.quantize(
        df["duration"], unit=unit_seconds
    )
    df["duration"] = pd.to_numeric(quantized_durations, errors="coerce")
    df["start_time"] = df["duration"].shift(1).fillna(0).cumsum()
    df["end_time"] = df["start_time"] + df["duration"]

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
    # add frame col
    df = _preprocess_add_frame_col(df)
    # to json
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(
        out_path,
        orient="records",
        indent=4,
        force_ascii=False,
    )


def preprocess_json(json_dir_path, mid_dir_path, out_path, parallel=False):
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            json_paths = list(get_files(json_dir_path, "json", sort=True))
            out_paths = [
                Path(out_path) / Path(json_path).name
                for json_path in json_paths
            ]
            mid_paths = get_files(mid_dir_path, "mid", sort=True)
            args = zip(json_paths, mid_paths, out_paths)
            p.starmap(_preprocess_json, args)
    else:
        json_paths = get_files(json_dir_path, "json", sort=True)
        mid_paths = get_files(mid_dir_path, "mid", sort=True)
        for json_path, mid_path in zip(json_paths, mid_paths):
            _preprocess_json(
                json_path, mid_path, Path(out_path) / Path(json_path).name
            )


def fill_time_gaps_save(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filled_time_gaps = fill_time_gaps_with_silence(data)
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filled_time_gaps, f, ensure_ascii=False, indent=4)
    print(f"save:\n{out_path}")


def verify_files_coherent(list1, list2):
    stems1 = {Path(f).stem for f in list1}
    stems2 = {Path(f).stem for f in list2}

    unique_stems_in_list1 = stems1.difference(stems2)
    unique_stems_in_list2 = stems2.difference(stems1)

    result_files_1 = []
    for file_path_str in list1:
        if Path(file_path_str).stem in unique_stems_in_list1:
            result_files_1.append(file_path_str)
    result_files_2 = []
    for file_path_str in list2:
        if Path(file_path_str).stem in unique_stems_in_list2:
            result_files_2.append(file_path_str)
    return result_files_1, result_files_2


def remove_abnormal_file(gv_dir):
    """
    gv 데이터셋 사전 준비:
    - 삭제
    """
    abnormal_files = (
        "SINGER_11_10TO29_NORMAL_MALE_DANCE_C0456.json",
        "SINGER_11_10TO29_NORMAL_MALE_DANCE_C0457.json",
        "SINGER_11_10TO29_NORMAL_MALE_DANCE_C0458.json",
        "SINGER_11_10TO29_NORMAL_MALE_DANCE_C0459.json",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0792.mid",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0793.mid",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0794.mid",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0795.mid",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0792.wav",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0793.wav",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0794.wav",
        "SINGER_19_10TO29_CLEAR_MALE_DANCE_C0795.wav",
    )
    deleted_files = []
    error_files = []
    target_filenames_set = set(abnormal_files)
    gv_dir = Path(gv_dir)
    for path_object in gv_dir.rglob("*"):
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


def _singer_id(filename):
    sid = re.findall(r"SINGER_\d\d", filename)[0]
    return int(sid[-2:]) + 44


def split_audio(y, sr, start_time, end_time, output_filename):
    start_sample = int(start_time * sr)  # Convert time to sample index
    end_sample = int(end_time * sr)

    chunk = y[start_sample:end_sample]  # Extract the segment
    soundfile.write(output_filename, chunk, sr)  # Save as WAV file


def verify_notes_sorted_by_time(json_filepath):
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes_data = data.get("notes")
    df = pd.DataFrame(notes_data)

    df["start_time"] = pd.to_numeric(df["start_time"])
    df["end_time"] = pd.to_numeric(df["end_time"])
    df["previous_end_time"] = df["end_time"].shift(1)
    overlapping_notes = df[
        df["previous_end_time"].notna()
        & (df["previous_end_time"] > df["start_time"])
    ]

    if not overlapping_notes.empty:
        return json_filepath, overlapping_notes
    return None, None


def _get_diffs(json_path):
    diffs = []
    _, result_df = verify_notes_sorted_by_time(json_path)
    if result_df is not None:
        for _, row in result_df.iterrows():
            diff = row["previous_end_time"] - row["start_time"]
            diffs.append(diff)
    return diffs


def verify_json_notes_sorted_by_time(dir_path, parallel=False):
    diffs = []
    if parallel:
        with mp.Pool(mp.cpu_count()) as p:
            diffs = p.map(
                _get_diffs,
                get_files(dir_path, "json"),
            )
            diffs = list(itertools.chain.from_iterable(diffs))
    else:
        for json_path in get_files(dir_path, "json"):
            path, result_df = verify_notes_sorted_by_time(json_path)
            if path is not None:
                for index, row in result_df.iterrows():
                    diffs.append(row["previous_end_time"] - row["start_time"])
    print_stat(diffs, verbose=True)


def split_json(json_filepath, split_json_filepath):
    split_json = split_json_by_silence(json_filepath, min_length=6)
    split_json_filepath = Path(split_json_filepath)
    split_json_filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(split_json_filepath, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=4, ensure_ascii=False)


def split_jsons(json_dirpath, split_json_dirpath):
    with mp.Pool(mp.cpu_count()) as p:
        json_paths = get_files(json_dirpath, "json")
        args = []
        for json_path in json_paths:
            json_path = Path(json_path)
            args.append((json_path, Path(split_json_dirpath) / json_path.name))
        p.starmap(split_json, args)


def preprocess_one(
    wav_path,
    json_path,
    out_path,
):
    wav_path = Path(wav_path)
    json_path = Path(json_path)
    if wav_path.stem != json_path.stem:
        raise ValueError
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    out_pitch_dir_path = Path(out_path) / "pitch"
    out_duration_dir_path = Path(out_path) / "duration"
    out_wavs_dir_path = Path(out_path) / "wav"
    out_pitch_dir_path.mkdir(exist_ok=True, parents=True)
    out_duration_dir_path.mkdir(exist_ok=True, parents=True)
    out_wavs_dir_path.mkdir(exist_ok=True, parents=True)

    metadata = ""
    filename = wav_path.stem
    y, sr = librosa.load(wav_path, sr=None)
    for i, chunk in enumerate(json_data):
        subfilename = f"{filename}_{i:02}"
        lyric = "".join([item["lyric"] for item in chunk["chunk"]])
        # lyric = " ".join([x for x in lyric.split()])
        metadata += f"{subfilename}|{lyric}|{_singer_id(filename)}|11|SV\n"
        np.save(
            f"{out_pitch_dir_path}/{subfilename}.npy",
            np.array([item["pitch"] for item in chunk["chunk"]]),
        )
        np.save(
            f"{out_duration_dir_path}/{subfilename}.npy",
            np.array([item["frames"] for item in chunk["chunk"]]),
        )
        split_audio(
            y,
            sr,
            start_time=chunk["chunk_info"]["start_time"],
            end_time=chunk["chunk_info"]["end_time"],
            output_filename=f"{out_wavs_dir_path}/{subfilename}.wav",
        )
    return metadata


def save_duration_pitch_metadata_split_audio(
    wav_dirpath,
    split_json_dirpath,
    preprocessed_gv_path,
):
    with mp.Pool(mp.cpu_count()) as p:
        wav_paths = get_files(wav_dirpath, "wav", sort=True)
        split_json_paths = get_files(split_json_dirpath, "json", sort=True)
        preprocessed_gv_paths = [
            preprocessed_gv_path for _ in range(len(list(split_json_paths)))
        ]
        args = zip(wav_paths, split_json_paths, preprocessed_gv_paths)
        metadata_list = p.starmap(preprocess_one, args)
    preprocessed_gv_path = Path(preprocessed_gv_path)
    preprocessed_gv_path.mkdir(exist_ok=True, parents=True)
    with open(
        f"{preprocessed_gv_path}/metadata.txt", "w", encoding="utf-8"
    ) as f:
        f.write("".join(metadata_list))
