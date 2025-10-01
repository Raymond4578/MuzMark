import shutil

import torch
from torch import nn
from omegaconf import DictConfig
from typing import Optional, Tuple

from .pitch_shift import pitch_shift
from third_party.raw_bench.raw_bench.attacks import AudioAttack

class MusicAttack(AudioAttack):
    def __init__(
        self,
        sr: int = 44100,
        datapath: dict = None,
        stft: nn.Module = None,
        mode: str = "test",
        config: DictConfig = None,
        ffmpeg4codecs: Optional[str] = None,
        mixing_train_filepath: Optional[str] = None,
        reverb_train_filepath: Optional[str] = None,
        delimiter: str = '|',
        single_attack: bool = True,
        device: str = "cuda:0"
    ):
        super().__init__(
            sr=sr,
            datapath=datapath,
            stft=stft,
            mode=mode,
            config=config,
            ffmpeg4codecs=ffmpeg4codecs,
            mixing_train_filepath=mixing_train_filepath,
            reverb_train_filepath=reverb_train_filepath,
            delimiter=delimiter,
            single_attack=single_attack,
            device=device
        )

    def apply_pitch_shift(
        self,
        x: torch.Tensor,
        semitones: Optional[int] = None
    ) -> Tuple[torch.Tensor, str, dict]:

        if self.mode == 'train' and semitones is None:
            semitones = torch.randint(
                low=-self.config.pitch_shift.max_semitones, high=self.config.pitch_shift.max_semitones + 1, size=(1,)
            ).item()

        elif self.mode in ['test', 'val'] and semitones is None:
            raise ValueError('semitones should be provided in val and test modes.')

        if semitones == 0:
            return x, "pitch_shift", {"semitones": semitones}

        output_tensor = pitch_shift(x, sr=self.sr, semitones=semitones)

        return output_tensor, "pitch_shift", {"semitones": semitones}

