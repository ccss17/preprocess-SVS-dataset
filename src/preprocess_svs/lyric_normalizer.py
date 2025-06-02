from typing import Dict, List, Tuple
import re

import hangul_dtw
from jamo import h2j, j2h


class LyricNormalizer:
    # TYPE_NO_CHANGE = 'no_change'
    TYPE_CHANGE = "change"
    TYPE_MERGE = "merge"
    TYPE_CHANGE_WITH_PITCH = "change_with_pitch_change"
    TYPE_MERGE_NO_PITCH = "merge_no_pitch_change"  # without pitch change
    TYPE_SILENCE = "silence"  # for spaces

    def __init__(self):
        """텍스트 정규화를 위한 클래스를 초기화합니다."""
        self.dtw = hangul_dtw.hangul_DTW
        self.vowel_set_map = {
            "ᅪ": "ᅡ",
            "ᅡ": "ᅡ",
            "ᅣ": "ᅡ",
            "ᅱ": "ᅵ",
            "ᅵ": "ᅵ",
            "ᅯ": "ᅥ",
            "ᅥ": "ᅥ",
            "ᅧ": "ᅥ",
            "ᅭ": "ᅩ",
            "ᅩ": "ᅩ",
            "ᅲ": "ᅮ",
            "ᅮ": "ᅮ",
            "ᅰ": "ᅦ",
            "ᅫ": "ᅦ",
            "ᅬ": "ᅦ",
            "ᅨ": "ᅦ",
            "ᅦ": "ᅦ",
            "ᅤ": "ᅦ",
            "ᅢ": "ᅢ",
            "ᅴ": "ᅵ",
            "ᅳ": "ᅳ",
        }

    def normalize_lyrics(
        self,
        gt_lyrics: str,
        raw_lyrics: str,
        pitch_sequence: List[int],
        duration_sequence: List[int],
        normalize_spaces: bool = True,
    ) -> Dict:
        """
        입력 텍스트를 정규화하여 반환합니다.

        Args:
            gt_text (str): 정답 텍스트
            raw_text (str): 원본 텍스트
            pitch_sequence (List[int]): raw_text의 각 음절에 대응하는 피치 값 리스트

        Returns:
            Dict: {
                'normalized_text': List[str],  # 정규화된 텍스트
                'normalized_pitch': List[int],  # 정규화된 피치
                'normalization_info': List[Dict]  # 정규화 정보
            }
        """
        _gt_lyrics = re.sub(r"[^ㄱ-ㅎ가-힣]+", "", gt_lyrics)
        _raw_lyrics = re.sub(r"[^ㄱ-ㅎ가-힣]+", "", raw_lyrics)
        gt_syllables = list(_gt_lyrics)
        raw_syllables = list(_raw_lyrics)

        # 1. 전처리 (remove silence from pitch, duration sequence)
        pitch_sequence, duration_sequence, space_indices = (
            self.remove_silence_from_pitch_and_duration(
                pitch_sequence, duration_sequence
            )
        )

        # 2. DTW를 통한 정렬
        _, _, _, syllable_mapping = self.dtw(
            gt_lyrics, raw_lyrics, space=False
        )

        # 3. 정규화 결과를 저장할 리스트들
        normalized_lyrics = []
        normalized_pitches = []
        normalized_durations = []
        normalization_infos = []
        current_idx = 0  # 현재 정규화된 텍스트의 인덱스

        # 4. syllable_mapping을 순회하며 정규화 수행
        for gt_idx, raw_indices in syllable_mapping.items():
            if len(raw_indices) == 1:
                # 1:1 매핑 처리
                lyrics, pitches, durations, infos = (
                    self.normalize_one_to_one_mapping(
                        gt_syllables,
                        gt_idx,
                        raw_indices,
                        pitch_sequence,
                        duration_sequence,
                        current_idx,
                    )
                )
            else:
                # 1:N 매핑 처리
                lyrics, pitches, durations, infos = (
                    self.normalize_one_to_many_mapping(
                        gt_syllables,
                        gt_idx,
                        raw_syllables,
                        raw_indices,
                        pitch_sequence,
                        duration_sequence,
                        current_idx,
                    )
                )

            normalized_lyrics.extend(lyrics)
            normalized_pitches.extend(pitches)
            normalized_durations.extend(durations)
            normalization_infos.extend(infos)
            current_idx += len(lyrics)  # 인덱스 업데이트

        # 5. raw text의 공백 위치 복원
        if normalize_spaces:
            insertion_points = self.find_space_insertion_points(
                raw_lyrics, normalization_infos, space_indices
            )
            (
                normalized_lyrics,
                normalized_pitches,
                normalized_durations,
                normalization_infos,
            ) = self.insert_spaces(
                normalized_lyrics,
                normalized_pitches,
                normalized_durations,
                normalization_infos,
                insertion_points,
            )

        return {
            "normalized_texts": normalized_lyrics,
            "normalized_pitches": normalized_pitches,
            "normalized_durations": normalized_durations,
            "normalization_infos": normalization_infos,
        }

    def find_space_insertion_points(
        self,
        raw_text: str,
        normalization_infos: List[Dict],
        space_indices: List[Tuple[int, int]],
    ) -> List[Dict]:
        """정규화된 텍스트에 공백을 삽입할 위치를 찾습니다."""
        insertion_points = []

        # 맵1: 원본 raw_text의 문자 인덱스 -> 공백 제거 raw_text의 문자 인덱스
        map1 = {}
        current_no_space_idx = 0
        for i, char_original in enumerate(raw_text):
            if not char_original.isspace():
                map1[i] = current_no_space_idx
                current_no_space_idx += 1

        # 맵2: 공백 제거 raw_text의 문자 인덱스 -> normalized_text의 문자 인덱스
        map2 = {}
        for info in normalization_infos:
            if info.get("type") == self.TYPE_SILENCE:
                continue
            current_normalized_idx_for_this_info = info["normalized_idx"]

            for raw_idx_ns in info["raw_indices"]:
                map2[raw_idx_ns] = current_normalized_idx_for_this_info

        # space_indices의 위치 정보를 사용하여 삽입 위치 결정
        for space_pos, duration in space_indices:
            idx1 = space_pos - 1
            idx2 = map1.get(idx1)
            idx3 = map2.get(idx2)
            insertion_point = idx3 + 1
            insertion_points.append(
                {
                    "position": insertion_point,
                    "raw_idx": space_pos,
                    "duration": duration,
                }
            )

        return sorted(
            insertion_points, key=lambda x: (x["position"], x["raw_idx"])
        )

    def insert_spaces(
        self,
        normalized_texts: List[str],
        normalized_pitches: List[int],
        normalized_durations: List[int],
        normalization_infos: List[Dict],
        insertion_points: List[Dict],
    ) -> Tuple[List[str], List[int], List[int], List[Dict]]:
        """정규화된 텍스트에 공백을 삽입합니다."""
        # 삽입 위치를 역순으로 정렬하여 인덱스 변화를 방지
        for point in reversed(insertion_points):
            pos = point["position"]
            normalized_texts.insert(pos, " ")
            normalized_pitches.insert(pos, 0)
            normalized_durations.insert(pos, point["duration"])

            # 공백에 대한 normalization_info 추가
            space_info = {
                "type": self.TYPE_SILENCE,
                "duration": point["duration"],
                "normalized_idx": pos,
            }
            normalization_infos.insert(pos, space_info)

            # 이후의 normalized_idx 업데이트
            for info in normalization_infos[pos + 1 :]:
                info["normalized_idx"] += 1

        return (
            normalized_texts,
            normalized_pitches,
            normalized_durations,
            normalization_infos,
        )

    def is_pitch_change(
        self, pitch_sequence: List[int], raw_indices: List[int]
    ) -> bool:
        """주어진 구간에 피치 변화가 있는지 확인합니다."""

        first_pitch = pitch_sequence[raw_indices[0]]
        for i in range(raw_indices[0] + 1, raw_indices[-1] + 1):
            if pitch_sequence[i] != first_pitch:
                return True
        return False

    def normalize_one_to_one_mapping(
        self,
        gt_syllables: List[str],
        gt_idx: int,
        raw_indices: List[int],
        pitch_sequence: List[int],
        duration_sequence: List[int],
        current_idx: int,
    ) -> Tuple[List[str], List[int], List[int], List[Dict]]:
        """1:1 매핑을 처리합니다."""
        gt_syllable = gt_syllables[gt_idx]

        normalized_info = {
            "type": self.TYPE_CHANGE,
            "normalized_syllable": gt_syllable,
            "normalized_duration": duration_sequence[raw_indices[0]],
            "gt_idx": gt_idx,
            "raw_indices": raw_indices,
            "normalized_idx": current_idx,
        }
        return (
            [gt_syllable],
            [pitch_sequence[raw_indices[0]]],
            [duration_sequence[raw_indices[0]]],
            [normalized_info],
        )

    def normalize_one_to_many_mapping(
        self,
        gt_syllables: List[str],
        gt_idx: int,
        raw_syllables: List[str],
        raw_indices: List[int],
        pitch_sequence: List[int],
        duration_sequence: List[int],
        current_idx: int,
    ) -> Tuple[List[str], List[int], List[int], List[Dict]]:
        """1:N 매핑을 처리합니다."""
        if not self.is_pitch_change(pitch_sequence, raw_indices):
            gt_syllable = gt_syllables[gt_idx]

            normalized_info = {
                "type": self.TYPE_MERGE,
                "normalized_syllable": gt_syllable,
                "normalized_duration": sum(
                    duration_sequence[raw_indices[0] : raw_indices[-1] + 1]
                ),
                "gt_idx": gt_idx,
                "raw_indices": raw_indices,
                "normalized_idx": current_idx,
            }

            normalized_duration = sum(
                duration_sequence[raw_indices[0] : raw_indices[-1] + 1]
            )

            return (
                [gt_syllable],
                [pitch_sequence[raw_indices[0]]],
                [normalized_duration],
                [normalized_info],
            )
        else:
            split_indices = self.split_indices_by_pitch_change(
                raw_indices, pitch_sequence
            )
            return self.normalize_one_to_many_mapping_by_pitch_change(
                gt_syllables,
                gt_idx,
                raw_syllables,
                split_indices,
                pitch_sequence,
                duration_sequence,
                current_idx,
            )

    def split_indices_by_pitch_change(
        self, raw_indices: List[int], pitch_sequence: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Example:
            raw_indices: [0, 1, 2, 3]
            pitch_sequence: [60, 60, 61, 61]
            return: [(0, 1), (2, 3)]
        """
        split_points = []
        for i in range(raw_indices[0], raw_indices[-1]):
            if pitch_sequence[i] != pitch_sequence[i + 1]:
                split_points.append(i)

        split_indices = []
        start = raw_indices[0]
        for point in split_points:
            split_indices.append((start, point))
            start = point + 1
        split_indices.append((start, raw_indices[-1]))

        return split_indices

    def normalize_one_to_many_mapping_by_pitch_change(
        self,
        gt_syllables: List[str],
        gt_idx: int,
        raw_syllables: List[str],
        split_indices: List[Tuple[int, int]],
        pitch_sequence: List[int],
        duration_sequence: List[int],
        current_idx: int,
    ) -> Tuple[List[str], List[int], List[int], List[Dict]]:
        # 각 청크 처리
        normalized_texts = []
        normalized_pitches = []
        normalized_durations = []
        normalization_infos = []

        gt_jamos = h2j(gt_syllables[gt_idx])  # GT 음절을 자모로 분리

        for i, (start, end) in enumerate(split_indices):
            if i == 0:  # 첫 번째 청크
                normalized_text = j2h(gt_jamos[0], gt_jamos[1])  # 초성 + 중성
            elif i == len(split_indices) - 1:  # 마지막 청크
                if len(gt_jamos) > 2:  # 종성이 있는 경우
                    normalized_text = j2h(
                        "ㅇ", self.vowel_set_map[gt_jamos[1]], gt_jamos[2]
                    )
                else:  # 종성이 없는 경우
                    normalized_text = j2h(
                        "ㅇ", self.vowel_set_map[gt_jamos[1]]
                    )
            else:  # 중간 청크
                normalized_text = j2h("ㅇ", self.vowel_set_map[gt_jamos[1]])

            normalized_texts.append(normalized_text)
            normalized_pitches.append(pitch_sequence[start])
            normalized_durations.append(
                sum(duration_sequence[start : end + 1])
            )
            normalization_infos.append(
                {
                    "type": self.TYPE_MERGE
                    if len(list(range(start, end + 1))) > 1
                    else self.TYPE_CHANGE,
                    "normalized_syllable": normalized_text,
                    "normalized_duration": sum(
                        duration_sequence[start : end + 1]
                    ),
                    "gt_idx": gt_idx,
                    "raw_indices": list(range(start, end + 1)),
                    "normalized_idx": current_idx + i,
                }
            )

        return (
            normalized_texts,
            normalized_pitches,
            normalized_durations,
            normalization_infos,
        )

    def remove_silence_from_pitch_and_duration(
        self, pitch_sequence, duration_sequence
    ):
        new_pitch_sequence = []
        new_duration_sequence = []
        space_indices = []

        if len(pitch_sequence) != len(duration_sequence):
            raise ValueError(
                "pitch_sequence and duration_sequence must have the same length"
            )
        else:
            for i in range(len(pitch_sequence)):
                if pitch_sequence[i] == 0 or pitch_sequence[i] == 96:
                    space_indices.append((i, duration_sequence[i]))
                else:
                    new_pitch_sequence.append(pitch_sequence[i])
                    new_duration_sequence.append(duration_sequence[i])
        return new_pitch_sequence, new_duration_sequence, space_indices
