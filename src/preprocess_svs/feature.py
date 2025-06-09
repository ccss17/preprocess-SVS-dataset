import torch
import parselmouth
import pyworld as pw
from scipy.signal import medfilt
import numpy as np
from librosa.filters import mel as librosa_mel_fn
# import torchcrepe


f0_extractors = ["harvest", "praat-parselmouth", "crepe"]
MAX_WAV_VALUE = 32768.0


def extract_f0(
    wav,
    sampling_rate,
    hop_length,
    extractor="praat-parselmouth",
    center=False,
    pitch_floor=None,
    voicing_threshold=0.7,
    octave_cost=0.02,
    octave_jump_cost=0.5,
):
    if not isinstance(wav, np.ndarray):
        wav = wav.squeeze().numpy()

    n_frames = int(len(wav) // hop_length)

    if extractor not in f0_extractors:
        raise Exception(f"Available f0 extractors are {f0_extractors}")

    #    print('[LOG] extracting f0 by using {}'.format(extractor))

    if extractor == "harvest":
        f0, t = pw.dio(
            wav.astype(np.float64),
            sampling_rate,
            frame_period=hop_length / sampling_rate * 1000,
        )
        f0 = f0[:n_frames]
    elif extractor == "praat-parselmouth":
        time_step = hop_length / sampling_rate
        snd = parselmouth.Sound(wav, sampling_frequency=sampling_rate)
        pitch_floor = (
            (3 / time_step / 4) if pitch_floor is None else pitch_floor
        )
        pitch_object = snd.to_pitch_ac(
            time_step=time_step,
            voicing_threshold=voicing_threshold,
            octave_cost=octave_cost,
            octave_jump_cost=octave_jump_cost,
        )
        f0 = pitch_object.selected_array["frequency"]
        f0 = medfilt(f0, kernel_size=5)

        # f0 = medfilt(f0, kernel_size=5)
        if center:
            f0 = np.pad(f0, [2, 2])
        else:
            pad_size = (n_frames - len(f0) + 1) // 2
            f0 = np.pad(
                f0,
                [[pad_size, n_frames - len(f0) - pad_size]],
                mode="constant",
            )
    # elif extractor == "crepe":
    #     model = "full"
    #     batch_size = 2048
    #     f0_min, f0_max = 71.0, 792.8
    #     wav, sampling_rate = torchcrepe.load.audio(wav_path)
    #     pitch, periodicity = torchcrepe.predict(
    #         wav,
    #         sampling_rate,
    #         hop_length,
    #         f0_min,
    #         f0_max,
    #         model,
    #         batch_size=batch_size,
    #         device="cuda",
    #         return_periodicity=True,
    #     )
    #     win_length = 4
    #     periodicity = torchcrepe.filter.median(periodicity, win_length)

    #     if pitch.shape[1] != periodicity.shape[1]:
    #         periodicity = periodicity[:, :-1]
    #     periodicity = torchcrepe.threshold.Silence(-90.0)(
    #         periodicity, wav, sampling_rate, hop_length
    #     )
    #     pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)
    #     pitch[np.isnan(pitch)] = 0
    #     f0 = np.array(pitch.squeeze())

    return f0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return torch.exp(x) / C


def spectral_normalize(magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output


def spectral_de_normalize(magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def extract_mel_spectrogram(
    y,
    sampling_rate,
    n_fft,
    win_length,
    hop_length,
    n_mel=80,
    fmin=55,
    fmax=8000,
    center=False,
    complex=False,
):
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)

    y = y.clamp(min=-1.0, max=1.0)
    global mel_basis, hann_window

    if fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        [int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)],
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=complex,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    energy = torch.norm(spec, p=2, dim=1)  # shape: (B, T)
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize(spec)

    return spec.squeeze(), energy.squeeze()


def extract_features(wav, sampling_rate, n_fft, win_length, hop_length):
    f0 = extract_f0(wav, sampling_rate, hop_length)
    mel, energy = extract_mel_spectrogram(
        wav, sampling_rate, n_fft, win_length, hop_length
    )
    return f0, mel, energy
