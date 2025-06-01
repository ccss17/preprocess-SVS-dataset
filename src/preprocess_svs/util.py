import pathlib
import multiprocessing as mp
import json
import itertools

import pandas as pd
import numpy as np
import midii


def get_files(dir_path, type, sort=False):
    paths = pathlib.Path(dir_path).rglob(f"*.{type}")
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
        with mp.Pool(mp.cpu_count()) as p:
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
