import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from typing import Union, Optional

from .timbre import Encoder as TimbreEncoder, Decoder as TimbreDecoder
from .wm_positioner_module import WatermarkPositioner

class MuzEncoder(nn.Module):
    def __init__(
        self,
        config,
        device: Union[torch.device, str] = "cuda:0"
    ):
        super(MuzEncoder, self).__init__()
        self.wm_model_config = config.wm_model
        self.muzmark_config = config.muzmark
        self.device = device

        # setup variables for easy use
        self.wm_model_ckpt_path = self.wm_model_config.model.test.model_path
        self.process_config = self.wm_model_config.process
        self.audio_sample_rate = self.wm_model_config.process.audio.sample_rate
        self.model_sample_rate = self.wm_model_config.sample_rate


        ###############################################################################################################
        # load model
        ###############################################################################################################
        if self.wm_model_config.model_type.lower() == "timbre":
            self.encoder = TimbreEncoder(
                process_config=self.wm_model_config.process,
                model_config=self.wm_model_config.model,
                msg_length=self.wm_model_config.message.len,
                device=self.device
            ).to(self.device)
            ckpt = torch.load(
                self.wm_model_ckpt_path, map_location=self.device, weights_only=True
            )
            self.encoder.load_state_dict(ckpt["encoder"])
            # logger.info(f"model <<{self.wm_model_ckpt_path}>> loaded.")
        else:
            raise ValueError(
                f"Unsupported watermarking model type: {self.wm_model_config.model_type.lower()}. "
                f'Supported types: "timbre".'
            )

        ###############################################################################################################
        # load watermark positioning module
        ###############################################################################################################
        if self.muzmark_config.wm_positioner.use:
            self.wm_positioner = WatermarkPositioner(
                sr=self.audio_sample_rate, hop_length=self.muzmark_config.wm_positioner.hop_length, device=self.device
            )
        else:
            self.wm_positioner = None

        ###############################################################################################################
        # prepare resample module if not using FSWE
        ###############################################################################################################
        if not self.muzmark_config.FSWE:
            self.downsampler = torchaudio.transforms.Resample(
                orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate
            ).to(self.device)
            self.upsampler = torchaudio.transforms.Resample(
                orig_freq=self.model_sample_rate, new_freq=self.audio_sample_rate
            ).to(self.device)
        else:
            self.downsampler, self.upsampler = None, None

    def preprocess_music(
        self,
        y: torch.Tensor
    ):
        """
        STFT audio
        ｜-> separate low frequency information and high frequency information
        ｜-> low frequency information transform back to audio + Resample to model sample rate

        Args:
            y: torch.Tensor
                Audio data tensor.

        Returns:
            y_low_model: torch.Tensor
                Low frequency resampled audio.
            spect_high_full: torch.Tensor
                High frequency spectrogram (same size as spectrogram).
            phase_high_full: torch.Tensor
                High frequency phase (same size as phase).
            cutoff_bin: int
                Index for where to cut the spectrogram and phase.
        """
        # STFT on input audio
        stft_c = torch.stft(
            y.squeeze(1),
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            return_complex=True
        )

        # get spectrogram and phase
        spect = torch.abs(stft_c)
        phase = torch.angle(stft_c)

        # calculate window dimension, get the binned frequency
        win_dim = int((self.process_config["mel"]["n_fft"] / 2) + 1)
        bin_freqs = np.arange(win_dim) * (self.audio_sample_rate / self.process_config["mel"]["n_fft"])
        # find the index where the frequency is greater than 11025 (for Timbre Model).
        cutoff_bin = np.searchsorted(bin_freqs, self.model_sample_rate / 2, side="right")

        # split the high and low frequency information
        # stft_c_low_full = stft_c.new_zeros(stft_c.shape)
        # stft_c_high_full = stft_c.new_zeros(stft_c.shape)
        #
        # stft_c_low_full[:, :cutoff_bin, :] = stft_c[:, :cutoff_bin, :]
        # stft_c_high_full[:, cutoff_bin:, :] = stft_c[:, cutoff_bin:, :]

        spect_low_full = spect.new_zeros(spect.shape)
        phase_low_full = phase.new_zeros(phase.shape)
        spect_high_full = spect.new_zeros(spect.shape)
        phase_high_full = phase.new_zeros(phase.shape)

        spect_low_full[:, :cutoff_bin, :] = spect[:, :cutoff_bin, :]
        phase_low_full[:, :cutoff_bin, :] = phase[:, :cutoff_bin, :]
        spect_high_full[:, cutoff_bin:, :] = spect[:, cutoff_bin:, :]
        phase_high_full[:, cutoff_bin:, :] = phase[:, cutoff_bin:, :]

        # Transfer spectrogram back to complex value
        stft_c_low = spect_low_full * torch.exp(1j * phase_low_full)

        # ISTFT to get audio for low frequency information
        y_low = torch.istft(
            stft_c_low,
            # stft_c_low_full,
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            length=y.shape[-1]
        )

        # Resample low frequency audio to model sample rate
        y_low_model = torchaudio.transforms.Resample(
            orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate
        ).to(self.device)(y_low).unsqueeze(1)

        return y_low_model, spect_high_full, phase_high_full, cutoff_bin
        # return y_low_model, stft_c_high_full, cutoff_bin

    def postprocess_music(
        self,
        y_low_wm_model: torch.Tensor,
        spect_high_full: torch.Tensor,
        phase_high_full: torch.Tensor,
        cutoff_bin: int):
        """

        Args:
            y_low_wm_model: torch.Tensor
                Low frequency watermarked audio.
            spect_high_full: torch.Tensor
                High frequency spectrogram from preprocessing (same size as spectrogram).
            phase_high_full: torch.Tensor
                High frequency phase from preprocessing (same size as phase).
            cutoff_bin: int
                Index for where to cut the spectrogram and phase.

        Returns:
            y_wm: torch.Tensor
                Watermarked full audio.
        """
        # resample the watermarked audio back to original sample rate
        y_low_wm = torchaudio.transforms.Resample(
            orig_freq=self.model_sample_rate, new_freq=self.audio_sample_rate
        ).to(self.device)(y_low_wm_model)
        # STFT on watermarked original sample rate audio
        stft_c_low_wm = torch.stft(
            y_low_wm.squeeze(1),
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            return_complex=True
        )

        # get spectrogram and phase for low frequency information
        spect_low_wm = torch.abs(stft_c_low_wm)
        phase_low_wm = torch.angle(stft_c_low_wm)

        # cut if spectrogram lengths are different
        spect_low_wm = spect_low_wm[..., :spect_high_full.size(-1)]
        phase_low_wm = phase_low_wm[..., :phase_high_full.size(-1)]

        # combine it back to the full frequency spectrogram
        spect_high_full[:, :cutoff_bin, :] = spect_low_wm[:, :cutoff_bin, :]
        phase_high_full[:, :cutoff_bin, :] = phase_low_wm[:, :cutoff_bin, :]

        # get comple STFT
        stft_c_wm = spect_high_full * torch.exp(1j * phase_high_full)

        # transform back to watermarked audio
        y_wm = torch.istft(
            stft_c_wm,
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            length=y_low_wm.shape[-1]
        ).unsqueeze(1)

        return y_wm

    def chunk_forward(self, x, msg, ratio = 1.0):
        ##############################
        # preprocess data for frequncy split
        ##############################
        if self.audio_sample_rate != self.model_sample_rate:
            x, spect_high_full, phase_high_full, cutoff_bin = self.preprocess_music(x)
            # x, stft_c_high_full, cutoff_bin = self.postprocess_music(x)
        else:
            x, spect_high_full, phase_high_full, cutoff_bin = x, None, None, None
            # x, stft_c_high_full, cutoff_bin = x, None, None
        # print(f"{spect_high_full.size()=}")

        ##############################
        # same as original
        ##############################
        if self.wm_model_config.model_type == "timbre":
            y, _, _, _ = self.encoder.test_forward(x, msg, ratio=ratio)
        else:
            raise ValueError(
                f"Unsupported watermarking model type: {self.wm_model_config.model_type.lower()}. "
                f'Supported types: "timbre".'
            )
        y = y[..., :x.size(2)]

        ##############################
        # postprocess data for frequncy combine
        ##############################
        if self.audio_sample_rate != self.model_sample_rate:
            y = self.postprocess_music(
                y, spect_high_full=spect_high_full, phase_high_full=phase_high_full, cutoff_bin=cutoff_bin
            )

        return y

    def normal_forward(self, x, msg, ratio = 1.0):
        # print("Encoder 使用 normal forward")
        if self.audio_sample_rate != self.model_sample_rate:
            x = self.downsampler(x)

        if self.wm_model_config.model_type == "timbre":
            y, _, _, _ = self.encoder.test_forward(x, msg, ratio=ratio)
        else:
            raise ValueError(
                f"Unsupported watermarking model type: {self.wm_model_config.model_type.lower()}. "
                f'Supported types: "timbre".'
            )
        y = y[..., :x.size(2)]

        if self.audio_sample_rate != self.model_sample_rate:
            y = self.upsampler(y)

        return y

    def get_position(self, wm_position_info):
        wm_start_times = wm_position_info["starts"][wm_position_info["positions"]]
        wm_duration_times = wm_position_info["durations"][wm_position_info["positions"]]
        wm_start_frames = (wm_start_times * self.audio_sample_rate).astype(int)
        wm_duration_frames = (wm_duration_times * self.audio_sample_rate).astype(int)
        return wm_start_frames, wm_duration_frames

    def full_forward(self, x, msg, ratio = 1.0):
        # first move to cpu to save memory
        x = x.to("cpu")
        msg = msg.to("cpu")

        # get watermark positions
        wm_position_dict = self.wm_positioner(x)

        # put all watermark chunks in each music into a full list for parallel processing
        batch_ls = list()
        batch_chunk_nums = list() # list of num of chunks in each music
        for b in range(x.size(0)):
            wm_position_info = wm_position_dict[b]
            wm_start_frames, wm_duration_frames = self.get_position(wm_position_info)
            batch_chunk_nums.append(wm_start_frames.size) # record num of wm chunks in this music
            for start, duration in zip(wm_start_frames, wm_duration_frames):
                batch_ls.append(x[b, :, start:(start + duration)])
        # combine these chunks into a single tensor with combination in batch
        batch_t_ls = [x.transpose(0, 1) for x in batch_ls]
        stacked_chunks_t = nn.utils.rnn.pad_sequence(batch_t_ls, batch_first=True, padding_value=0)
        stacked_chunks = stacked_chunks_t.transpose(2, 1)

        # repeat msg in batch for each music
        msg_ls = list()
        for i, r in enumerate(batch_chunk_nums):
            msg_ls.append(msg[i:(i + 1)].expand(r, -1, -1))
        msg_for_chunks = torch.cat(msg_ls, dim=0)

        # form dataloader to prepare for inference
        data_loader = DataLoader(
            TensorDataset(stacked_chunks, msg_for_chunks),
            batch_size=self.muzmark_config.wm_positioner.batch_size, shuffle=False
        )
        # do inference
        stacked_chunks_wm_ls = list()
        for chunks, msgs in data_loader:
            if self.muzmark_config.FSWE:
                chunks_wm = self.chunk_forward(chunks.to(self.device), msgs.to(self.device), ratio=ratio)
            else:
                chunks_wm = self.normal_forward(chunks.to(self.device), msgs.to(self.device), ratio=ratio)
            stacked_chunks_wm_ls.append(chunks_wm)
        stacked_chunks_wm = torch.cat(stacked_chunks_wm_ls, dim=0) # recombine into a full tensor in batch

        # get the watermarked chunks for each music
        chunks_wm_ls = torch.split(stacked_chunks_wm, batch_chunk_nums, dim=0)

        # arrange output music
        y = x.clone().to(self.device)
        for b in range(y.size(0)):
            wm_position_info = wm_position_dict[b]
            wm_start_frames, wm_duration_frames = self.get_position(wm_position_info)

            # get watermarked chunks for this music
            chunks_wm = chunks_wm_ls[b]
            # replace these chunks with watermarked chunks
            for i, position_pair in enumerate(zip(wm_start_frames, wm_duration_frames)):
                start, duration = position_pair
                y[b, :, start:(start + duration)] = chunks_wm[i, :, :duration]

        return y, wm_position_dict

    def forward(self, x, msg, ratio = 1.0):
        if self.muzmark_config.wm_positioner.use:
            return self.full_forward(x, msg, ratio=ratio)
        else:
            return self.chunk_forward(x, msg, ratio=ratio), None




class MuzDecoder(nn.Module):
    def __init__(
        self,
        config,
        device: Union[torch.device, str] = "cuda:0"
    ):
        super(MuzDecoder, self).__init__()

        self.wm_model_config = config.wm_model
        self.muzmark_config = config.muzmark
        self.device = device

        # setup variables for easy use
        self.wm_model_ckpt_path = self.wm_model_config.model.test.model_path
        self.process_config = self.wm_model_config.process
        self.audio_sample_rate = self.wm_model_config.process.audio.sample_rate
        self.model_sample_rate = self.wm_model_config.sample_rate

        ###############################################################################################################
        # load model
        ###############################################################################################################
        if self.wm_model_config.model_type.lower() == "timbre":
            self.decoder = TimbreDecoder(
                process_config=self.wm_model_config.process,
                model_config=self.wm_model_config.model,
                msg_length=self.wm_model_config.message.len,
                vocoder_config=self.wm_model_config.vocoder.vocoder_config,
                vocoder_checkpoint=self.wm_model_config.vocoder.vocoder_checkpoint,
                device=self.device
            ).to(self.device)
            ckpt = torch.load(
                self.wm_model_ckpt_path, map_location=self.device, weights_only=True
            )
            self.decoder.load_state_dict(ckpt["decoder"], strict=False)

        ###############################################################################################################
        # load watermark positioning module
        ###############################################################################################################
        if self.muzmark_config.wm_positioner.use:
            self.wm_positioner = WatermarkPositioner(
                sr=self.audio_sample_rate, hop_length=self.muzmark_config.wm_positioner.hop_length,
                device=self.device
            )
        else:
            self.wm_positioner = None

        ###############################################################################################################
        # prepare resample module if not using FSWE
        ###############################################################################################################
        if not self.muzmark_config.FSWE:
            self.downsampler = torchaudio.transforms.Resample(
                orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate
            ).to(self.device)
        else:
            self.downsampler = None

    def preprocess_music(
        self,
        y_wm: torch.Tensor
    ):
        """
        STFT audio
        ｜-> separate low frequency information and high frequency information
        ｜-> low frequency information transform back to audio + Resample to model sample rate

        Args:
            y_wm: torch.Tensor
                Watermarked audio data tensor.

        Returns:
            y_low_model: torch.Tensor
                Low frequency resampled audio.
        """
        # STFT on input audio
        stft_c_wm = torch.stft(
            y_wm.squeeze(1),
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            return_complex=True
        )

        # get spectrogram and phase
        spect_wm = torch.abs(stft_c_wm)
        phase_wm = torch.angle(stft_c_wm)

        # calculate window dimension, get the binned frequency
        win_dim = int((self.process_config["mel"]["n_fft"] / 2) + 1)
        bin_freqs = np.arange(win_dim) * (self.audio_sample_rate / self.process_config["mel"]["n_fft"])
        # find the index where the frequency is greater than 11025 (for Timbre Model).
        cutoff_bin = np.searchsorted(bin_freqs, self.model_sample_rate / 2, side="right")

        # split the high and low frequency information
        spect_wm_low_full = spect_wm.new_zeros(spect_wm.shape)
        phase_wm_low_full = phase_wm.new_zeros(phase_wm.shape)

        spect_wm_low_full[:, :cutoff_bin, :] = spect_wm[:, :cutoff_bin, :]
        phase_wm_low_full[:, :cutoff_bin, :] = phase_wm[:, :cutoff_bin, :]

        # Transfer spectrogram back to complex value
        stft_c_wm_low = spect_wm_low_full * torch.exp(1j * phase_wm_low_full)

        # ISTFT to get audio for low frequency information
        y_wm_low = torch.istft(
            stft_c_wm_low,
            n_fft=self.process_config.mel.n_fft,
            hop_length=self.process_config.mel.hop_length,
            win_length=self.process_config.mel.win_length,
            window=torch.hann_window(self.process_config.mel.n_fft).to(self.device),
            center=True,
            length=y_wm.shape[-1]
        )

        # Resample low frequency audio to model sample rate
        y_wm_low_model = torchaudio.transforms.Resample(
            orig_freq=self.audio_sample_rate, new_freq=self.model_sample_rate
        ).to(self.device)(y_wm_low).unsqueeze(1)

        return y_wm_low_model

    def chunk_forward(self, y):
        ##############################
        # preprocess data for frequency split
        ##############################
        if self.audio_sample_rate != self.model_sample_rate:
            y = self.preprocess_music(y)

        ##############################
        # same as original
        ##############################
        if self.wm_model_config.model_type == "timbre":
            msg = self.decoder.test_forward(y)
        else:
            raise ValueError(
                f"Unsupported watermarking model type: {self.wm_model_config.model_type.lower()}. "
                f'Supported types: "timbre".'
            )
        return msg

    def normal_forward(self, y):
        # print("Decoder 使用 normal forward")
        if self.audio_sample_rate != self.model_sample_rate:
            y = self.downsampler(y)

        if self.wm_model_config.model_type == "timbre":
            return self.decoder.test_forward(y)
        else:
            raise ValueError(
                f"Unsupported watermarking model type: {self.wm_model_config.model_type.lower()}. "
                f'Supported types: "timbre".'
            )

    # def get_position(self, wm_position_info):
    #     wm_start_times = wm_position_info["starts"][wm_position_info["positions"]]
    #     wm_duration_times = wm_position_info["durations"][wm_position_info["positions"]]
    #     wm_start_frames = (wm_start_times * self.audio_sample_rate).astype(int)
    #     wm_duration_frames = (wm_duration_times * self.audio_sample_rate).astype(int)
    #     return wm_start_frames, wm_duration_frames
    def get_position(self, wm_position_info): # 改成了会返回所有小节，因为后面已经设计了排除不对的小节
        start_times = wm_position_info["starts"]
        duration_times = wm_position_info["durations"]
        start_frames = (start_times * self.audio_sample_rate).astype(int)
        duration_frames = (duration_times * self.audio_sample_rate).astype(int)
        return start_frames, duration_frames

    def full_forward(self, y, start_bits: torch.Tensor):
        # first move to cpu to save memory
        y = y.to("cpu")

        # get watermark positions
        wm_position_dict = self.wm_positioner(y)

        # put all watermark chunks in each music into a full list for parallel processing
        batch_ls = list()
        batch_chunk_nums = list()  # list of num of chunks in each music
        for b in range(y.size(0)):
            position_info = wm_position_dict[b]
            start_frames, duration_frames = self.get_position(position_info)
            batch_chunk_nums.append(start_frames.size)  # record num of wm chunks in this music
            for start, duration in zip(start_frames, duration_frames):
                batch_ls.append(y[b, :, start:(start + duration)])
        # combine these chunks into a single tensor with combination in batch
        batch_t_ls = [x.transpose(0, 1) for x in batch_ls]
        stacked_chunks_t = nn.utils.rnn.pad_sequence(batch_t_ls, batch_first=True, padding_value=0)
        stacked_chunks = stacked_chunks_t.transpose(2, 1)

        # form dataloader to prepare for inference
        data_loader = DataLoader(
            TensorDataset(stacked_chunks),
            batch_size=self.muzmark_config.wm_positioner.batch_size, shuffle=False
        )
        # do inference
        stacked_msgs_ls = list()
        for (chunks, ) in data_loader:
            if self.muzmark_config.FSWE:
                msgs = self.chunk_forward(chunks.to(self.device))
            else:
                msgs = self.normal_forward(chunks.to(self.device))
            stacked_msgs_ls.append(msgs)
        stacked_msgs = torch.cat(stacked_msgs_ls, dim=0)  # recombine into a full tensor in batch

        # get the msg for each music
        msgs_ls = torch.split(stacked_msgs, batch_chunk_nums, dim=0)

        ###############################################################
        # 试图让不对的message直接不参与计算
        aggregated_msgs_ls = list() # 记录每一首music最终算好的msg是什么
        preamble_oa_results = list() # initialise 一个list来记录 preamble oa准确率
        preamble_bw_results = list()  # initialise 一个list来记录 preamble bw准确率
        for batch_idx, msgs in enumerate(msgs_ls): # 循环，看每个music decode出来的msg的情况
            valid_msgs_ls = list() # initialize一个list，用来存放通过preamble检查的msg
            preamble_scores = list() # 准备一个list，根据preamble记录这条message的得分。这个只在完全没有preamble对上的情况下使用
            ###########
            current_preamble_oa = list() # initialise 一个list记录这一条msg的preamble对上还是没对上
            current_preamble_bw = list()  # initialise 一个list记录这一条msg的preamble正确率
            ###########
            for ch in range(msgs.size(0)): # 循环检查每一个decode出来的msg
                msg_head_three_logits = msgs[ch, 0, :start_bits.size(-1)] # 截取msg的preamble的logits
                msg_head_three = 2 * (msg_head_three_logits >= 0).float() - 1 # 转换成1 或者 -1
                if start_bits[batch_idx, 0, :].equal(msg_head_three): # 如果preamble能对上
                    valid_msgs_ls.append(msgs[ch:(ch + 1), ...]) # 记下这个msg
                    ###########
                    current_preamble_oa.append(1)
                    current_preamble_bw.append(1)
                    ###########
                else:
                    # 计算 bitwise accuracy，给decoded的preamble和真实的preamble
                    bw_acc = start_bits[batch_idx, 0, :].eq(msg_head_three).float().mean().item()
                    preamble_scores.append(bw_acc)
                    ###########
                    current_preamble_oa.append(0)
                    current_preamble_bw.append(bw_acc)
                    ###########
                    continue # 就不记了
            # 记录一下preamble准确率oa的结果和bw的结果
            preamble_oa_results.append(np.mean(current_preamble_oa))
            preamble_bw_results.append(np.mean(current_preamble_bw))
            if len(valid_msgs_ls) > 0: # 如果至少一个对上
                valid_msgs = torch.cat(valid_msgs_ls, dim=0) # 把valid的msgs并成torch
                aggregated_msgs_ls.append(valid_msgs.sum(dim=0, keepdim=True)) # 把valide的msgs加起来变成最后的msg
            else: # 如果一个没对上
                if set(preamble_scores) == 0: # 如果全部没一个preamble是的对的
                    aggregated_msgs_ls.append(msgs.sum(dim=0, keepdim=True)) # 直接按照权重所有加起来
                else:
                    preamble_scores_pt = torch.tensor(preamble_scores, dtype=torch.float32).to(self.device) # 转换成tensor
                    aggregated_msgs_ls.append(
                        (msgs * preamble_scores_pt[:, None, None]).sum(dim=0, keepdim=True)
                    ) # 记录weighted

        # aggregated_msgs_ls = [msgs.sum(dim=0, keepdim=True) for msgs in msgs_ls]
        ###############################################################

        msgs = torch.cat(aggregated_msgs_ls, dim=0)

        return msgs, wm_position_dict, {"overall": preamble_oa_results, "bitwise": preamble_bw_results}

    def forward(self, y, start_bits: Optional[torch.Tensor] = None):
        if self.muzmark_config.wm_positioner.use:
            return self.full_forward(y, start_bits)
        else:
            return self.chunk_forward(y), None, None






