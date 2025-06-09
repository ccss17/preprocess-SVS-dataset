from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path
import unicodedata

from jamo import h2j, j2h
import numpy as np
import whisper

import hangul_dtw


class ErrorChecker:
    def __init__(self):
        pass

    def check_errors(self, transcribed_text: str):
        """오인식 여부를 확인하고 오류 목록을 반환합니다.

        Args:
            transcribed_text (str): Whisper로 추론된 텍스트
            original_lyrics (str): 원본 가사
            filename (str): 파일 이름

        Returns:
            Tuple[bool, List[str]]: (오인식 여부, 오류 메시지 리스트)
        """
        try:
            is_error = False

            # 1. 숫자 포함 여부 확인
            if any(char.isdigit() for char in transcribed_text):
                is_error = True

            # 2. 한글 외 언어 포함 여부 확인 (공백과 특수문자 제외)
            non_hangul_chars = [
                char
                for char in transcribed_text
                if not self._is_hangul(char)
                and not char.isspace()
                and not char in ".,!?()[]{}'\"-_"
            ]
            if non_hangul_chars:
                is_error = True

            # 3. 멜리스마 확인
            if self._is_melisma(transcribed_text):
                is_error = True

            # 4. 한글 가사 없는 경우
            if not any(self._is_hangul(char) for char in transcribed_text):
                is_error = True

            # 5. '자막 제공' 텍스트 포함 여부
            if "자막 제공" in transcribed_text:
                is_error = True

            return is_error
        except IndexError as e:
            print(f"처리중인 텍스트: {transcribed_text}")

            return True

    def _is_hangul(self, char: str) -> bool:
        """문자가 한글인지 확인합니다."""
        return "HANGUL" in unicodedata.name(char)

    def _is_hangul_syllable(self, char: str) -> bool:
        """문자가 완성형 한글 음절인지 확인합니다."""
        return 0xAC00 <= ord(char) <= 0xD7A3

    def _is_melisma(self, text: str) -> bool:
        """멜리스마 여부를 확인합니다."""
        # 1. 한 음절로만 구성된 경우
        if len(text.strip()) == 1 and self._is_hangul_syllable(text.strip()):
            return True

        # 2. 초성 'ㅇ' 또는 'ㅎ'이며 중성과 종성이 동일한 3개 이상의 음절이 연속되는 경우
        text = text.replace(" ", "")
        if len(text) < 3:
            return False

        for i in range(len(text) - 2):
            if not all(
                self._is_hangul_syllable(char) for char in text[i : i + 3]
            ):
                continue

            # 각 음절을 자모로 분해
            syllables = [
                self._decompose_hangul(char) for char in text[i : i + 3]
            ]

            # 초성이 'ㅇ' 또는 'ㅎ'인지 확인
            if not all(s[0] in ["ㅇ", "ㅎ"] for s in syllables):
                continue

            # 중성과 종성이 모두 동일한지 확인
            if all(s[1] == syllables[0][1] for s in syllables) and all(
                s[2] == syllables[0][2] for s in syllables
            ):
                return True

        return False

    def _decompose_hangul(self, char: str) -> Tuple[str, str, str]:
        """한글 음절을 초성, 중성, 종성으로 분해합니다."""
        if not self._is_hangul_syllable(char):
            return ("", "", "")

        code = ord(char) - 0xAC00
        jong = code % 28
        jung = (code // 28) % 21
        cho = code // 28 // 21

        CHOSUNG = [
            "ㄱ",
            "ㄲ",
            "ㄴ",
            "ㄷ",
            "ㄸ",
            "ㄹ",
            "ㅁ",
            "ㅂ",
            "ㅃ",
            "ㅅ",
            "ㅆ",
            "ㅇ",
            "ㅈ",
            "ㅉ",
            "ㅊ",
            "ㅋ",
            "ㅌ",
            "ㅍ",
            "ㅎ",
        ]
        JUNGSUNG = [
            "ㅏ",
            "ㅐ",
            "ㅑ",
            "ㅒ",
            "ㅓ",
            "ㅔ",
            "ㅕ",
            "ㅖ",
            "ㅗ",
            "ㅘ",
            "ㅙ",
            "ㅚ",
            "ㅛ",
            "ㅜ",
            "ㅝ",
            "ㅞ",
            "ㅟ",
            "ㅠ",
            "ㅡ",
            "ㅢ",
            "ㅣ",
        ]
        JONGSUNG = [
            "",
            "ㄱ",
            "ㄲ",
            "ㄳ",
            "ㄴ",
            "ㄵ",
            "ㄶ",
            "ㄷ",
            "ㄹ",
            "ㄺ",
            "ㄻ",
            "ㄼ",
            "ㄽ",
            "ㄾ",
            "ㄿ",
            "ㅀ",
            "ㅁ",
            "ㅂ",
            "ㅄ",
            "ㅅ",
            "ㅆ",
            "ㅇ",
            "ㅈ",
            "ㅊ",
            "ㅋ",
            "ㅌ",
            "ㅍ",
            "ㅎ",
        ]

        return (CHOSUNG[cho], JUNGSUNG[jung], JONGSUNG[jong])


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
        try:
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
        except Exception as e:
            print(f"error1 in normalize_lyric: {e}")
            return None

        # 2. DTW를 통한 정렬
        try:
            _, _, _, syllable_mapping = self.dtw(
                gt_lyrics, raw_lyrics, space=True
            )
        except Exception as e:
            print(f"Error in DTW: {e}")
            return None

        # 3. 정규화 결과를 저장할 리스트들
        normalized_lyrics = []
        normalized_pitches = []
        normalized_durations = []
        normalization_infos = []
        current_idx = 0  # 현재 정규화된 텍스트의 인덱스

        # 4. syllable_mapping을 순회하며 정규화 수행
        try:
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
        except Exception as e:
            print(f"Error2 in normalize_lyric: {e}")
            return None

        # 5. raw text의 공백 위치 복원
        try:
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
        except Exception as e:
            print(f"Error3 in normalize_lyric: {e}")
            return None

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


class SVS_Preprocessor:
    def __init__(
        self,
        base_path: Path,
        model_name: str = "large-v3",
        device: str = "cuda",
        language: str = "ko",
    ):
        self.base_path = Path(base_path)
        self.model_name = model_name
        self.device = device
        self.language = language

        # Initialize paths
        self.metadata_path = self.base_path / "metadata.txt"
        self.normalized_metadata_path = (
            self.base_path / "normalized_metadata.txt"
        )
        self.wav_path = self.base_path / "wav"
        self.pitch_path = self.base_path / "pitch"
        self.duration_path = self.base_path / "duration"

        # Initialize components
        self.lyric_normalizer = LyricNormalizer()
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model."""
        self.model = whisper.load_model(self.model_name, device=self.device)

    def get_file_paths(self, filename_stem: str) -> Tuple[Path, Path, Path]:
        """Get the paths for wav, pitch, and duration files."""
        return (
            self.wav_path / f"{filename_stem}.wav",
            self.pitch_path / f"{filename_stem}.npy",
            self.duration_path / f"{filename_stem}.npy",
        )

    def transcribe_audio(self, wav_filepath: Path) -> Tuple[str, float]:
        """Transcribe audio file using Whisper model."""
        result = self.model.transcribe(
            str(wav_filepath), language=self.language
        )
        return result["text"]

    def load_sequences(
        self, pitch_filepath: Path, duration_filepath: Path
    ) -> Tuple[List, List]:
        """Load pitch and duration sequences from files."""
        pitch_sequence = np.load(pitch_filepath).tolist()
        duration_sequence = np.load(duration_filepath).tolist()
        return pitch_sequence, duration_sequence

    def save_normalized_sequences(
        self,
        duration_filepath: Path,
        pitch_filepath: Path,
        normalized_durations: List,
        normalized_pitches: List,
    ) -> None:
        """Save normalized sequences to files."""
        np.save(duration_filepath, np.array(normalized_durations))
        np.save(pitch_filepath, np.array(normalized_pitches))

    def normalize_lyrics_and_sequences(
        self,
        gt_text: str,
        original_lyrics: str,
        pitch_sequence: List,
        duration_sequence: List,
    ):
        """Normalize lyrics and corresponding sequences."""

        normalization_result = self.lyric_normalizer.normalize_lyrics(
            gt_lyrics=gt_text,
            raw_lyrics=original_lyrics,
            pitch_sequence=pitch_sequence,
            duration_sequence=duration_sequence,
            normalize_spaces=True,
        )

        if normalization_result is None:
            return None, None, None

        normalized_lyrics = "".join(
            normalization_result.get("normalized_texts", [])
        )
        normalized_durations = normalization_result.get(
            "normalized_durations", []
        )
        normalized_pitches = normalization_result.get("normalized_pitches", [])

        return normalized_lyrics, normalized_durations, normalized_pitches

    def process_metadata_line(self, line: str) -> Optional[str]:
        """Process a single line from metadata file."""
        line = line.strip()
        if not line:
            print("No line")
            return None

        parts = line.split("|")
        if len(parts) < 2:
            print(f"Skipping malformed line: {line}")
            return None

        original_filename_stem = parts[0]
        original_lyrics = parts[1]
        other_columns = parts[2:]

        wav_filepath, pitch_filepath, duration_filepath = self.get_file_paths(
            original_filename_stem
        )

        # print("----------------------------------------------")
        # print(f"Processing: {original_filename_stem}")
        # print( f"  Original Lyrics: '{original_lyrics}, {len(original_lyrics)}'")

        # Transcribe audio
        try:
            gt_text = self.transcribe_audio(wav_filepath)
            # print(f"  STT Result: '{gt_text}'")
        except Exception as e:
            print(f"wav issue in {original_filename_stem}")
            return None

        error_checker = ErrorChecker()
        is_error = error_checker.check_errors(gt_text)
        if is_error:
            self.log_error(
                original_filename_stem, "W", original_lyrics, gt_text
            )
            # print("is_error!")
            return None

        # Load sequences
        pitch_sequence, duration_sequence = self.load_sequences(
            pitch_filepath, duration_filepath
        )
        # print( f"  Original Pitch Sequence: {pitch_sequence}, {len(pitch_sequence)}")
        # print( f"  Original Duration Sequence: {duration_sequence}, {len(duration_sequence)}")

        # Normalize
        try:
            normalized_lyrics, normalized_durations, normalized_pitches = (
                self.normalize_lyrics_and_sequences(
                    gt_text,
                    original_lyrics,
                    pitch_sequence,
                    duration_sequence,
                )
            )

            if normalized_lyrics is None:
                self.log_error(
                    original_filename_stem, "D", original_lyrics, gt_text
                )
                print("error in normalized!")
                return None

            # print( f"  Normalized Lyrics: '{normalized_lyrics}', {len(normalized_lyrics)}")
            # print( f"  Normalized Durations: {normalized_durations}, {len(normalized_durations)}")
            # print( f"  Normalized Pitch Sequence: {normalized_pitches}, {len(normalized_pitches)}\n")

            # Save normalized sequences
            self.save_normalized_sequences(
                duration_filepath,
                pitch_filepath,
                normalized_durations,
                normalized_pitches,
            )

            # Prepare new metadata line
            new_line_parts = [original_filename_stem, normalized_lyrics]
            if other_columns:
                new_line_parts.extend(other_columns)

            return "|".join(new_line_parts)
        except Exception as e:
            self.log_error(
                original_filename_stem, "D", original_lyrics, gt_text
            )
            print("error in norm seqeunce")
            print(f"Error message: {e}")
            return None

    def log_error(
        self,
        filename: str,
        error_type: str,
        original_lyrics: str = "",
        gt_text: str = "",
    ) -> None:
        """
        오류를 error_list.txt에 기록합니다.

        Args:
            filename: 파일 이름
            error_type: 오류 유형 ('W' for Whisper error, 'D' for DTW calculation error)
            original_lyrics: 원본 가사 (DTW 에러의 경우)
            gt_text: Whisper STT 결과 (DTW 에러의 경우)
        """
        error_log_path = self.base_path / "error_list.txt"

        # 오류 로그 파일이 없으면 헤더 추가
        if not error_log_path.exists():
            with open(error_log_path, "w", encoding="utf-8") as f:
                f.write("filename|error_type|original_lyrics|gt_text\n")

        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(f"{filename}|{error_type}|{original_lyrics}|{gt_text}\n")

    def verify_dataset_consistency(self) -> Dict[str, List[str]]:
        """
        데이터셋의 일관성을 검증합니다.
        각 파일의 lyric length, pitch 개수, duration 개수가 모두 일치하는지 확인합니다.

        Returns:
            Dict[str, List[str]]: 검증 결과를 담은 딕셔너리
                - 'errors': 오류가 있는 파일들의 목록
                - 'warnings': 경고가 있는 파일들의 목록
        """
        print("\n=== Starting Dataset Consistency Verification ===")

        errors = []
        warnings = []

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 2:
                    continue

                filename_stem = parts[0]
                lyrics = parts[1]

                # 파일 경로 가져오기
                wav_filepath, pitch_filepath, duration_filepath = (
                    self.get_file_paths(filename_stem)
                )

                # 파일 존재 여부 확인
                if not all(
                    p.exists()
                    for p in [wav_filepath, pitch_filepath, duration_filepath]
                ):
                    errors.append(
                        f"{filename_stem}: One or more required files are missing"
                    )
                    continue

                try:
                    # 데이터 로드
                    pitch_sequence = np.load(pitch_filepath)
                    duration_sequence = np.load(duration_filepath)

                    # 길이 검증
                    lyric_length = len(lyrics)
                    pitch_length = len(pitch_sequence)
                    duration_length = len(duration_sequence)

                    # 모든 길이가 일치하는지 확인
                    if not (lyric_length == pitch_length == duration_length):
                        error_msg = (
                            f"{filename_stem}: Length mismatch - "
                            f"lyrics({lyric_length}), "
                            f"pitch({pitch_length}), "
                            f"duration({duration_length})"
                        )
                        errors.append(error_msg)

                    # 추가 검증: pitch와 duration이 음수나 비정상적인 값을 가지지 않는지
                    if np.any(pitch_sequence < 0):
                        warnings.append(
                            f"{filename_stem}: Contains negative pitch values"
                        )
                    if np.any(duration_sequence <= 0):
                        errors.append(
                            f"{filename_stem}: Contains non-positive duration values"
                        )

                except Exception as e:
                    errors.append(
                        f"{filename_stem}: Error during verification - {str(e)}"
                    )

        # 검증 결과 출력
        print("\n=== Verification Results ===")
        if errors:
            print("\nErrors found:")
            for error in errors:
                print(f"- {error}")
        else:
            print("\nNo errors found!")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"- {warning}")
        else:
            print("\nNo warnings!")

        return {"errors": errors, "warnings": warnings}

    def process_all_files(self) -> None:
        """Process all files in the metadata file."""
        if not self.model:
            self.load_model()

        processed_lines = []

        try:
            with (
                open(self.metadata_path, "r", encoding="utf-8") as f_in,
                open(
                    self.normalized_metadata_path, "w", encoding="utf-8"
                ) as f_out,
            ):
                for line in f_in:
                    processed_line = self.process_metadata_line(line)
                    # print(f"processed_line: {processed_line}")
                    if processed_line:
                        # processed_lines.append(processed_line)
                        f_out.write(processed_line + "\n")
                        f_out.flush()
                print("running step 4 Done.\n")
        except Exception as e:
            print(f"An error occurred: {e}")
        # if processed_lines:
        #    print(f"\nWriting normalized metadata to: {self.metadata_path}")
        #    with open(self.metadata_path, "w", encoding="utf-8") as f_out:
        #        for line in processed_lines:
        #            f_out.write(line + "\n")
        #    print("Done.")

        # 모든 처리가 완료된 후 검증 실행
        print("\nStarting dataset verification...")
        verification_results = self.verify_dataset_consistency()

        # 검증 결과에 따라 적절한 메시지 출력
        if verification_results["errors"]:
            print("\nWARNING: Dataset verification found errors!")
            print(
                "Please check the errors above and fix them before proceeding."
            )
        else:
            print("\nDataset verification completed successfully!")

        # else:
        #    print("No lines were processed to write to metadata.txt.")


# Usage
if __name__ == "__main__":
    preprocessor = SVS_Preprocessor(
        base_path="preprocessed_gv",  # or preprocessed_mssv
        model_name="large-v3",
        device="cuda",
        language="ko",
    )
    preprocessor.process_all_files()
