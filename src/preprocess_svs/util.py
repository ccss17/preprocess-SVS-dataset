from pathlib import Path
from multiprocessing import Pool, cpu_count
import json
from collections import defaultdict
from scipy.io.wavfile import read

import pandas as pd
import numpy as np
from g2pk2 import G2p
import librosa
import soundfile
import midii

from .feature import extract_features


def split_json_by_silence(json_path, min_length=6):
    def append_chunk(result, chunk):
        result.append(
            {
                "chunk_info": {
                    "start_time": chunk[0]["start_time"],
                    "end_time": chunk[-1]["end_time"],
                    "duration": sum(item["duration"] for item in chunk),
                },
                "chunk": chunk,
            }
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = []
    chunk = []
    chunk_length = 0
    for note in data:
        if note["lyric"] == " " and chunk_length > min_length:
            append_chunk(result, chunk)
            chunk = []
            chunk_length = 0
        else:
            chunk.append(note)
            chunk_length += note["duration"]
    if chunk:
        append_chunk(result, chunk)
    return result


def _preprocess_merge_silence(df):
    output_notes = []
    i = 0
    n = len(df)
    while i < n:
        current_row = df.iloc[i]  # Pandas Series
        if current_row["lyric"] == " ":
            merged_start_time = current_row["start_time"]
            merged_end_time = current_row["end_time"]

            j = i + 1
            while j < n and df.iloc[j]["lyric"] == " ":
                merged_end_time = df.iloc[j][
                    "end_time"
                ]  # 마지막 공백의 end_time으로 업데이트
                j += 1

            merged_item = {
                "start_time": merged_start_time,
                "end_time": merged_end_time,
                "pitch": 0,
                "lyric": " ",
                "duration": merged_end_time - merged_start_time,
            }
            output_notes.append(merged_item)
            i = j  # 병합된 블록 다음으로 인덱스 이동
        else:
            non_space_item = {
                "start_time": current_row["start_time"],
                "end_time": current_row["end_time"],
                "pitch": current_row["pitch"],
                "lyric": current_row["lyric"],
                "duration": current_row["duration"],
            }
            output_notes.append(non_space_item)
            i += 1
    df = pd.DataFrame(output_notes)
    return df


def _preprocess_remove_short_silence(df, threshold=0.3):
    processed_notes = []
    absorbed_time = 0.0

    for i in range(len(df)):
        current_note_s = df.iloc[i]
        if (
            current_note_s["lyric"] == " "
            and current_note_s["duration"] < threshold
        ):
            absorbed_time += current_note_s["duration"]
            continue
        else:
            note_to_add = current_note_s.to_dict()
            if absorbed_time > 0:
                note_to_add["start_time"] -= absorbed_time
                note_to_add["duration"] = (
                    note_to_add["end_time"] - note_to_add["start_time"]
                )
                absorbed_time = 0.0
            processed_notes.append(note_to_add)

    df = pd.DataFrame(processed_notes)
    return df


def _preprocess_silence_pitch_zero(df):
    df.loc[df["lyric"] == " ", "pitch"] = 0
    return df


def _preprocess_remove_front_back_silence(df):
    is_valid_lyric = df["lyric"] != " "
    valid_indices = df.index[is_valid_lyric].tolist()
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    df = df.iloc[first_valid_idx : last_valid_idx + 1].reset_index(drop=True)
    return df


def _preprocess_add_frame_col(df):
    df["frames"] = midii.second2frame(seconds=df["duration"].values)
    return df


def _preprocess_sort_by_start_time(df):
    df = df.sort_values(by="start_time").reset_index(drop=True)
    return df


def get_files(dir_path, type, sort=False):
    paths = Path(dir_path).rglob(f"*.{type}")
    if sort:
        return sorted(paths, key=lambda p: p.stem)
    else:
        return paths


def calculate_top_tempo_percentage(ranked_tempos):
    top_tempo_count = ranked_tempos[0][1]
    total_count = sum(count for tempo, count in ranked_tempos)
    percentage = (top_tempo_count / total_count) * 100
    return percentage


def get_midi_dominated_tempo(midi_path):
    mid = midii.MidiFile(midi_path, convert_1_to_0=True)
    tempo_rank = mid.tempo_rank()
    return calculate_top_tempo_percentage(tempo_rank)


def print_stat(data, verbose=False):
    if len(data) == 0:
        return
    data = np.asarray(data)
    min_np = np.min(data)
    max_np = np.max(data)
    mean_np = np.mean(data)
    std_np_population = np.std(data)  # 모집단 표준편차
    std_np_sample = np.std(data, ddof=1)  # 표본 표준편차

    print(f"len: {len(data)}")
    print(f"min: {min_np}")
    print(f"max: {max_np}")
    print(f"mean: {mean_np}")
    print(f"population stddev: {std_np_population:.2f}")
    print(f"sample stddev(ddof=1): {std_np_sample:.2f}")
    if verbose:
        data = np.round(data).astype(np.int_)
        unique_elements, counts = np.unique(data, return_counts=True)
        counted_data = list(zip(unique_elements, counts))
        sorted_counted_data = sorted(
            counted_data, key=lambda item: (-item[1], -item[0])
        )
        for element, count in sorted_counted_data:
            print(f"{element}: {count}")


def tempo_statistics(dir_path, parallel=False, verbose=False):
    if parallel:
        with Pool(cpu_count()) as p:
            dominated_tempo_ratio_list = p.map(
                get_midi_dominated_tempo,
                get_files(dir_path, "mid"),
            )
    else:
        dominated_tempo_ratio_list = []
        for midi_path in get_files(dir_path, "mid"):
            dominated_tempo_ratio_list.append(
                get_midi_dominated_tempo(midi_path)
            )
    data_np = np.array(dominated_tempo_ratio_list)
    print_stat(data_np, verbose=verbose)
    print(f"99 >= {np.sum(data_np >= 99)}")
    print(f"95 >= {np.sum(data_np >= 95)}")
    print(f"90 >= {np.sum(data_np >= 90)}")
    print(f"80 >= {np.sum(data_np >= 80)}")
    print(f"50 >= {np.sum(data_np >= 0)}")


g2p_instance_worker = None


def init_worker():
    """Initialize a G2p instance for each worker process."""
    global g2p_instance_worker
    g2p_instance_worker = G2p()
    # print(f"Worker {os.getpid()} G2p_instance_worker initialized.") # 디버깅용


def apply_g2p_to_text(text_to_process):
    """Applies g2p to a single text string using the worker's G2p instance."""
    if g2p_instance_worker is None:
        # Fallback, though initializer should prevent this in Pool
        temp_g2p = G2p()
        return temp_g2p(text_to_process)
    return g2p_instance_worker(text_to_process)


def g2p_metadata(file_path):
    df = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=["ID", "Text", "Num1", "Num2", "Category"],
    )

    # 원본 텍스트와 길이를 저장
    original_texts = df["Text"].astype(str).tolist()
    df["Original_Length"] = df["Text"].astype(str).apply(len)

    # 멀티프로세싱 Pool을 사용하여 g2p 병렬 적용
    num_processes = cpu_count()
    g2p_processed_texts = []
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        g2p_processed_texts = pool.map(apply_g2p_to_text, original_texts)
    df["g2p_Text"] = g2p_processed_texts
    df["g2p_Length"] = df["g2p_Text"].astype(str).apply(len)

    # 원본에 비해 길이가 줄어든 row를 식별
    # g2p_Length >= Original_Length 인 경우만 유지
    rows_to_keep_mask = df["g2p_Length"] >= df["Original_Length"]

    # print(f"\n원본 데이터 행 개수: {len(df)}")
    df_updated = df[
        rows_to_keep_mask
    ].copy()  # .copy()를 사용하여 SettingWithCopyWarning 방지
    # print(f"길이 조건 필터링 후 행 개수: {len(df_updated)}")

    # 필터링된 DataFrame의 'Text' 컬럼을 g2p 변환 결과로 업데이트
    if not df_updated.empty:
        df_updated.loc[:, "Text"] = df_updated["g2p_Text"]

    # 최종적으로 필요한 컬럼만 선택 (중간 과정 컬럼 제외)
    final_columns = ["ID", "Text", "Num1", "Num2", "Category"]
    df_final_result = df_updated[final_columns]
    df_final_result.to_csv(
        file_path, sep="|", header=False, index=False, encoding="utf-8"
    )


def split_audio(y, sr, start_time, end_time, output_filename):
    start_sample = int(start_time * sr)  # Convert time to sample index
    end_sample = int(end_time * sr)

    chunk = y[start_sample:end_sample]  # Extract the segment
    soundfile.write(output_filename, chunk, sr)  # Save as WAV file


def preprocess_one(
    wav_path, json_path, out_path, singer_id_from_filepath=lambda: 0
):
    wav_path = Path(wav_path)
    json_path = Path(json_path)
    if wav_path.stem != json_path.stem:
        raise ValueError
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    out_path = Path(out_path)

    out_pitch_dir_path = out_path / "pitch"
    out_duration_dir_path = out_path / "duration"
    out_wavs_dir_path = out_path / "wav"
    out_pitch_dir_path.mkdir(exist_ok=True, parents=True)
    out_duration_dir_path.mkdir(exist_ok=True, parents=True)
    out_wavs_dir_path.mkdir(exist_ok=True, parents=True)

    metadata = ""
    filename = wav_path.stem
    try:
        y, sr = librosa.load(wav_path, sr=None)
    except FileNotFoundError:
        return
    for i, chunk in enumerate(json_data):
        subfilename = f"{filename}_{i:02}"
        lyric = "".join([item["lyric"] for item in chunk["chunk"]])
        metadata += f"{subfilename}|{lyric}|{singer_id_from_filepath(filename)}|11|SV\n"
        pitch_subfile = f"{out_pitch_dir_path}/{subfilename}.npy"
        pitch_subfile = Path(pitch_subfile)
        duration_subfile = f"{out_duration_dir_path}/{subfilename}.npy"
        duration_subfile = Path(duration_subfile)
        wav_subfile = f"{out_wavs_dir_path}/{subfilename}.wav"
        wav_subfile = Path(wav_subfile)
        if not pitch_subfile.exists():
            np.save(
                pitch_subfile,
                np.array([item["pitch"] for item in chunk["chunk"]]),
            )
        if not duration_subfile.exists():
            np.save(
                duration_subfile,
                np.array([item["frames"] for item in chunk["chunk"]]),
            )
        if not wav_subfile.exists():
            split_audio(
                y,
                sr,
                start_time=chunk["chunk_info"]["start_time"],
                end_time=chunk["chunk_info"]["end_time"],
                output_filename=wav_subfile,
            )
    return metadata


def save_duration_pitch_metadata_split_audio(
    wav_dirpath,
    split_json_dirpath,
    preprocessed_path,
    singer_id_from_filepath=lambda: 0,
):
    wav_dirpath = Path(wav_dirpath)
    split_json_paths = get_files(split_json_dirpath, "json", sort=True)
    args = []
    for split_json_path in split_json_paths:
        arg = (
            wav_dirpath / Path(split_json_path).with_suffix(".wav").name,
            split_json_path,
            preprocessed_path,
            singer_id_from_filepath,
        )
        args.append(arg)

    with Pool(cpu_count()) as p:
        metadata_list = p.starmap(preprocess_one, args)

    preprocessed_path = Path(preprocessed_path)
    preprocessed_path.mkdir(exist_ok=True, parents=True)
    with open(f"{preprocessed_path}/metadata.txt", "w", encoding="utf-8") as f:
        f.write("".join(metadata_list))


def split_json(json_filepath, split_json_filepath):
    split_json = split_json_by_silence(json_filepath, min_length=6)
    split_json_filepath = Path(split_json_filepath)
    if split_json_filepath.exists():
        return
    split_json_filepath.parent.mkdir(exist_ok=True, parents=True)
    with open(split_json_filepath, "w", encoding="utf-8") as f:
        json.dump(split_json, f, indent=4, ensure_ascii=False)


def split_jsons(json_dirpath, split_json_dirpath):
    json_paths = get_files(json_dirpath, "json")
    split_json_dirpath = Path(split_json_dirpath)
    args = []
    for json_path in json_paths:
        json_path = Path(json_path)
        args.append((json_path, split_json_dirpath / json_path.name))
    with Pool(cpu_count()) as p:
        p.starmap(split_json, args)


def find_exclusive_two_type_files(type1, type2, dir1_path, dir2_path):
    dir1_path = Path(dir1_path)
    dir2_path = Path(dir2_path)
    type1_files_by_basename = defaultdict(list)
    type2_files_by_basename = defaultdict(list)

    for type1_file_path in dir1_path.rglob(f"*.{type1}"):
        type1_files_by_basename[type1_file_path.stem].append(type1_file_path)
    for type2_file_path in dir2_path.rglob(f"*.{type2}"):
        type2_files_by_basename[type2_file_path.stem].append(type2_file_path)

    type1_basenames = set(type1_files_by_basename.keys())
    type2_basenames = set(type2_files_by_basename.keys())
    exclusive_basenames = type1_basenames.symmetric_difference(type2_basenames)

    result_file_paths = []
    for basename in exclusive_basenames:
        if basename in type1_basenames:
            result_file_paths.extend(type2_files_by_basename[basename])
        elif basename in type2_basenames:
            result_file_paths.extend(type2_files_by_basename[basename])
        else:
            raise NotImplementedError

    return result_file_paths


#
# 임시
#
def read_wav(wav_path, normalize=False):
    _, wav = read(wav_path)
    wav = wav.astype(np.float32)

    if normalize:
        wav = wav / 32768.0

    return wav


def save_f0_mel_energy(wav_path, out_path):
    extracted_wav = read_wav(wav_path, normalize=True)
    f0, mel, energy = extract_features(
        extracted_wav,
        sampling_rate=midii.DEFAULT_SAMPLING_RATE,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
    )

    out_path = Path(out_path)
    wav_path = Path(wav_path)

    out_f0_dir_path = out_path / "f0"
    out_mel_dir_path = out_path / "mel"
    out_energy_dir_path = out_path / "energe"
    out_f0_dir_path.mkdir(exist_ok=True, parents=True)
    out_mel_dir_path.mkdir(exist_ok=True, parents=True)
    out_energy_dir_path.mkdir(exist_ok=True, parents=True)

    f0_subfile = f"{out_f0_dir_path}/{wav_path.stem}.npy"
    f0_subfile = Path(f0_subfile)
    if not f0_subfile.exists():
        np.save(f0_subfile, f0)

    mel_subfile = f"{out_mel_dir_path}/{wav_path.stem}.npy"
    mel_subfile = Path(mel_subfile)
    if not mel_subfile.exists():
        np.save(mel_subfile, mel)

    energy_subfile = f"{out_energy_dir_path}/{wav_path.stem}.npy"
    energy_subfile = Path(energy_subfile)
    if not energy_subfile.exists():
        np.save(energy_subfile, energy)


def save_all_f0_mel_energy(wav_dir_path, out_path):
    args = []
    for wav_path in get_files(wav_dir_path, "wav"):
        args.append((wav_path, out_path))
    with Pool(cpu_count()) as p:
        p.starmap(save_f0_mel_energy, args)
