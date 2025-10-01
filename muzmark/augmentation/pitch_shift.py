import torch
import torchaudio

def pitch_shift(
    wav_tensor: torch.Tensor,
    sr: int,
    semitones: int = 0
):
    device = wav_tensor.device
    return torchaudio.transforms.PitchShift(sample_rate=sr, n_steps=semitones).to(device)(wav_tensor)



if __name__ == "__main__":
    import os

    # set up device
    device = "cuda:2" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

    music_name = "Kataomoi"
    if music_name == "Deadman":
        audio_path = f"../../data/test_samples/{music_name}.flac"
    else:
        audio_path = f"../../data/test_samples/{music_name}.mp3"

    sig, sr = torchaudio.load(audio_path)
    sig = sig.unsqueeze(0).to(device)
    print(f"{sig.size()=}")

    semitones = 5

    new_sig = pitch_shift(wav_tensor=sig, sr=sr, semitones=semitones)
    print(f"{new_sig.size()=}")

    os.makedirs("../../outputs/pitch_shift", exist_ok=True)
    torchaudio.save(
        f"../../outputs/pitch_shift/{music_name}_pitch_shift_{semitones}.wav",
        new_sig.to("cpu").squeeze(0), sr
    )