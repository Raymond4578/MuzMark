import numpy as np
import librosa
import torch
import torch.nn as nn
import torchaudio
from typing import Union
from beat_this.inference import Audio2Beats



class WatermarkPositioner(nn.Module):
    def __init__(
        self,
        sr: int = 48000,
        hop_length: int = 512,
        device: Union[torch.device, str] = "cuda:0"
    ):
        super(WatermarkPositioner, self).__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.device = device
        # initialise BeatThis for beat detection
        self.audio2beats = Audio2Beats(checkpoint_path="final0", device=self.device, dbn=False)

    def detect_beats(
        self,
        signal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detects beats in a music audio signal.

        Args:
            signal: np.ndarray
                A 1D np array for music data.

        Returns:
            beats: np.ndarray
                A 1D np array for beats (including downbeats).
            downbeats: np.ndarray
                A 1D np array for downbeats.
        """
        # convert dimension to fit Beat This model
        signal = signal[:, np.newaxis]
        return self.audio2beats(signal, sr=self.sr)

    def clean_beats_downbeats(
        self,
        beats: np.ndarray,
        downbeats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cleans beats and downbeats by removing too crowded beats.

        Args:
            beats: np.ndarray
                A 1D np array for beats (including downbeats).
            downbeats: np.ndarray
                A 1D np array for downbeats.

        Returns:
            clean_beats: np.ndarray
                A 1D np array for cleaned beats (including downbeats).
            clean_downbeats: np.ndarray
                A 1D np array for cleaned downbeats.
        """
        # clean beats
        beat_durations = np.diff(beats)
        # filter out very small beats and very large beats
        # max tempo:300 -> min duration = 60 / 300 = 0.2
        valid_beat_duration_idx = beat_durations > 0.2
        valid_beat_idx = np.append(True, valid_beat_duration_idx)

        # clean downbeats
        downbeat_durations = np.diff(downbeats)
        # minimially assume 3 beats a bar
        valid_downbeat_duration_idx = downbeat_durations > (3 * 0.2)
        valid_downbeat_idx = np.append(True, valid_downbeat_duration_idx)

        return beats[valid_beat_idx], downbeats[valid_downbeat_idx]

    def estimate_tempo(
        self,
        beats: np.ndarray
    ) -> float:
        """
        Estimates the tempo of a music audio signal.

        Args:
            beats: np.ndarray
                A 1D np array for beats (including downbeats).

        Returns:
            tempo: float
                The estimated tempo of the music.
        """
        if beats.size <= 1:
            raise ValueError("At least two beats are required to compute tempo.")

        beat_durations = np.diff(beats)
        tempo = 60.0 / float(np.median(beat_durations))
        # restrict to a normal tempo
        while tempo > 130.0: tempo /= 2.0
        while tempo < 50.0: tempo *= 2.0

        return tempo

    def detect_onsets(
        self,
        signal: np.ndarray,
        tempo: float
    ) -> np.ndarray:
        """
        Detects onset positions in a music audio signal.

        Args:
            signal: np.ndarray
                A 1D np array for music data.

        Returns:
            onset_peak_times: np.ndarray
                A 1D np array for onset peak times.
        """
        onset_env = librosa.onset.onset_strength(y=signal, sr=self.sr, hop_length=self.hop_length, aggregate=np.median)

        # get indicator for wait
        good_wait = int(0.5 * (60.0 / tempo) * (self.sr / tempo))

        # detect onset times
        onset_peaks = librosa.onset.onset_detect(
            onset_envelope=librosa.util.normalize(onset_env), sr=self.sr, hop_length=self.hop_length,
            backtrack=True,
            pre_max=20, post_max=20,
            pre_avg=100, post_avg=100,
            delta=0.15,  # 阈值偏移；越大越严格
            wait=max(10, good_wait)  # 最小间隔（帧）；越大越稀
        )
        onset_peak_times = librosa.frames_to_time(onset_peaks, sr=self.sr, hop_length=self.hop_length)
        return onset_peak_times

    def find_onset_bars(
        self,
        bar_starts: np.ndarray,
        onset_times: np.ndarray
    ) -> np.ndarray:
        # initilise output string
        onset_per_bar = np.zeros_like(bar_starts)
        # initialise pointer for onset times
        j = 0
        for i in range(len(bar_starts)):
            # identify bar start and bar end time
            start = bar_starts[i]
            end = bar_starts[i + 1] if (i + 1) < len(bar_starts) else np.inf

            has = False
            # skip all onset times before the start of this bar
            while j < len(onset_times) and onset_times[j] < start: j += 1
            # go through all onset times in this bar
            while j < len(onset_times) and onset_times[j] < end:
                onset_per_bar[i] += 1
                j += 1

        return onset_per_bar

    def balance_positioning(
        self,
        onset_per_bar: np.ndarray
    ) -> np.ndarray:
        onset_bars = onset_per_bar.astype(bool)
        # initialise balanced onset bars
        balanced_onset_bars = np.copy(onset_bars)

        i = 0
        while i + 2 < len(onset_bars):
            first, second = onset_bars[i:i + 2]
            # first look at two
            if first != second:
                i += 2
            else:
                third = onset_bars[i + 2]
                if first != third:
                    i += 2
                else:
                    balanced_onset_bars[i + 1] = not balanced_onset_bars[i + 1]
                    i += 3
        return balanced_onset_bars

    # 这是一个主要的处理函数，上面的函数都是这个函数中的小步骤
    def arrange_position(self, signal):
        # convert to numpy for librosa
        signal_np = signal.to("cpu").numpy().astype(np.float64)

        # estimate beat times
        beats, downbeats = self.detect_beats(signal_np)
        beats, downbeats = self.clean_beats_downbeats(beats, downbeats)
        # form a bar frame by downbeats
        bar_starts = np.append(0.0, downbeats) if downbeats[0] != 0.0 else downbeats
        bar_durations = np.diff(np.append(bar_starts, (signal_np.size / self.sr)))

        # estimate onset times
        tempo = self.estimate_tempo(beats)
        onset_times = self.detect_onsets(signal_np, tempo)

        # get onset bars
        onset_per_bar = self.find_onset_bars(bar_starts, onset_times)
        balanced_positions = self.balance_positioning(onset_per_bar)

        return balanced_positions, bar_starts, bar_durations, beats, downbeats, onset_times, tempo

    def forward(self, signals, timepoints: bool = False):
        # check if the input tensor has desired shape
        if signals.size(1) != 1 or signals.dim() != 3:
            raise ValueError(
                f"Expected input tensor of shape [batch_size, 1, signal_length] "
                f"(mono audio), but got shape {tuple(signals.shape)}."
            )

        wm_position_dict = dict()
        # loop through all batches to get start position and duration for watermarking
        for b in range(signals.size(0)):
            signal = signals[b, 0, :]
            balanced_positions, bar_starts, bar_durations, beats, downbeats, onset_times, tempo = self.arrange_position(signal)
            wm_position_dict[b] = {
                "positions": balanced_positions,
                "starts": bar_starts,
                "durations": bar_durations,
                "tempo": tempo
            }
            if timepoints:
                wm_position_dict[b] = {
                    "beats": beats,
                    "downbeats": downbeats,
                    "onset_times": onset_times
                }

        return wm_position_dict


if __name__ == "__main__":
    import soundfile as sf

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    music_name = "MoonInOnesCup"
    orig_y, sr = torchaudio.load(
        f"../../data/test_samples/{music_name}.flac" if music_name == "Deadman" else f"../../data/test_samples/{music_name}.mp3"
    )
    if orig_y.size(0) > 1:
        y = torch.mean(orig_y, dim=0, keepdim=True)
    else:
        y = orig_y
    y = y.unsqueeze(0).repeat(1, 1, 1).to(device)

    wm_position = WatermarkPositioner(sr=sr, hop_length=512, device=device)

    wm_position_dict = wm_position(y, timepoints=True)
    # print(f"{wm_position_dict=}")

    def listen_beats(orig_signal, sr, beats, downbeats):
        # remove beats which are downbeats
        beats = np.array([b for b in beats if b not in downbeats])

        # generate click sounds on beat time
        downbeat_clicks = librosa.clicks(
            times=downbeats, sr=sr, length=orig_signal.size(1), click_freq=2000.0, click_duration=0.05
        )
        beat_clicks = librosa.clicks(
            times=beats, sr=sr, length=orig_signal.size(1), click_freq=1500.0, click_duration=0.05
        )

        # repeat clicks for stereo
        downbeat_clicks_stereo = np.repeat(downbeat_clicks[:, None], 2, axis=1)
        beat_clicks_stereo = np.repeat(beat_clicks[:, None], 2, axis=1)

        # normalize for output
        beats_mix = orig_signal.permute(1, 0).numpy() + downbeat_clicks_stereo + beat_clicks_stereo
        beats_mix = librosa.util.normalize(beats_mix)

        return beats_mix

    beats_mix = listen_beats(
        orig_signal=orig_y, sr=sr, beats=wm_position_dict[0]["beats"], downbeats=wm_position_dict[0]["downbeats"]
    )
    sf.write(f"../../test_beats_audios/{music_name}_with_beats_BeatThis.wav", beats_mix, sr)

    def listen_onsets(orig_signal, sr, onset_times):
        onset_clicks = librosa.clicks(
            times=onset_times, sr=sr, length=orig_signal.size(1), click_freq=1500.0, click_duration=0.05
        )

        onset_mix = orig_signal.permute(1, 0).numpy() + np.repeat(onset_clicks[:, None], 2, axis=1)
        onset_mix = librosa.util.normalize(onset_mix)

        return onset_mix

    onset_mix = listen_onsets(orig_signal=orig_y, sr=sr, onset_times=wm_position_dict[0]["onset_times"])
    sf.write(f"../../test_beats_audios/{music_name}_with_onsets.wav", onset_mix, sr)
