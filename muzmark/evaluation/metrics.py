import io
import warnings
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from typing import Union
import torchaudio
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from frechet_audio_distance import FrechetAudioDistance
from torcheval.metrics import FrechetAudioDistance
from mel_cepstral_distance import compare_audio_files

class MusicEvaluator(nn.Module):
    '''
    Music quality evaluation
    '''

    def __init__(
        self,
        sr: int = 44100,
        device: Union[torch.device, str] = "cuda:0"
    ):
        super().__init__()
        self.sr = sr
        self.device = device

        self.fad = FrechetAudioDistance.with_vggish(device=self.device)

    def calculate_SISNR(
        self,
        orig_signal: torch.Tensor,
        wm_signal: torch.Tensor
    ) -> float:
        return float(scale_invariant_signal_noise_ratio(wm_signal, orig_signal))

    def calculate_FAD(
        self,
        orig_signal: torch.Tensor,
        wm_signal: torch.Tensor
    ) -> float:
        # check sample rate, only do 16000
        # Warn the user and resample to 16000
        if self.sr != 16000:
            warnings.warn(
                f"The input music sample rate is {self.sr} Hz, but FAD (Fréchet Audio Distance) method only accept sample rate of 16000 Hz. The music will be resampled to 16000 hz for FAD evaluation.",
                UserWarning
            )

            resampler = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=16000).to(self.device)
            orig_signal = resampler(orig_signal)
            wm_signal = resampler(wm_signal)

        # calculate FAD
        self.fad.reset()
        self.fad.update(orig_signal, wm_signal)
        return float(self.fad.compute().item())

        # # transfer to numpy
        # orig_signal = orig_signal.to("cpu").view(-1).numpy().astype(np.float32)
        # wm_signal = wm_signal.to("cpu").view(-1).numpy().astype(np.float32)
        #
        # fad = FrechetAudioDistance(
        #     model_name="vggish",
        #     sample_rate=16000,
        #     use_pca=False,
        #     use_activation=False,
        #     verbose=False
        # )
        #
        # # get embedding
        # orig_embed = fad.get_embeddings([orig_signal], 16000)
        # wm_embed = fad.get_embeddings([wm_signal], 16000)
        #
        # # get statistics (mu & sigma)
        # orig_mu, orig_sigma = fad.calculate_embd_statistics(orig_embed)
        # wm_mu, wm_sigma = fad.calculate_embd_statistics(wm_embed)
        #
        # fad_score = fad.calculate_frechet_distance(orig_mu, orig_sigma, wm_mu, wm_sigma)
        #
        # return float(fad_score)

    def calculate_MCD(
        self,
        orig_signal: torch.Tensor,
        wm_signal: torch.Tensor
    ) -> float:
        # get data ready (copied from Raw_Bench)

        # In-memory paths for the audio files instead of writing to disk
        orig_path = io.BytesIO()
        wm_path = io.BytesIO()

        # compare_audio_files reads from the disk, so we need to save the audio files first.
        # This is not optimal, but let's keep it for now.
        sf.write(orig_path, orig_signal.to("cpu").view(-1), samplerate=self.sr, format='wav')
        sf.write(wm_path, wm_signal.to("cpu").view(-1), samplerate=self.sr, format='wav')

        # 指针回到0
        for buffer in [orig_path, wm_path]:
            buffer.seek(0)

        n_fft_ms = 2048 * 1000 / self.sr
        hop_length_ms = n_fft_ms / 4
        # get MCD value
        mcd_wm, _ = compare_audio_files(
            orig_path, wm_path,
            sample_rate=self.sr, n_fft=n_fft_ms, win_len=n_fft_ms, hop_len=hop_length_ms
        )

        return float(mcd_wm)

    def forward(
        self,
        orig_signal: torch.Tensor,
        wm_signal: torch.Tensor,
        metric_types: list = None
    ) -> dict:
        if metric_types == None:
            metric_types = ["SISNR", "FAD", "MCD"]
        # check validity of metric types
        if not isinstance(metric_types, list) or len(metric_types) == 0:
            raise ValueError('"metric_types" should be a non-empty list.')

        # prepare calculate evaluation
        output_dict = dict()
        for metric in set(metric_types):
            # check validity of metrics
            if metric not in ["SISNR", "FAD", "MCD"]:
                raise ValueError(f'The input metric is "{metric}", while the metric type should be either "SISNR", "FAD" or "MCD".')
            # get the calculation function
            calculator = getattr(self, f"calculate_{metric}", None)
            output_dict[metric] = calculator(orig_signal, wm_signal)

        return output_dict



if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    evaluator = MusicEvaluator(sr=44100, device=device)

    torch.manual_seed(79)
    orig_signal = torch.randn(1, 44100 * 10).to(device)
    wm_signal = torch.randn(1, 44100 * 10).to(device)

    print(evaluator(orig_signal, wm_signal, metric_types=["MCD"]))