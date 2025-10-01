

import os
import json
import shutil
import pandas as pd
import torch
import torchaudio
from omegaconf import OmegaConf, DictConfig
from typing import Union
from pathlib import Path
from qqdm import qqdm
from torch.utils.data import DataLoader
from loguru import logger

from third_party.raw_bench.raw_bench.utils import init_worker
from third_party.raw_bench.raw_bench.custom_stft import STFT
from muzmark.augmentation import MusicAttack
import muzmark.models.timbre.timbre as timbre
from muzmark.dataloader import MusicDataset
from muzmark.evaluation import MusicEvaluator
from muzmark.evaluation import check_wm_position_acc
import muzmark.models.muzmark as muzmark
from muzmark.models.ldpc_module import LDPC






# Rewrite from SolverTimbre from raw_bench
class BenchmarkPipeline(object):
    def __init__(
        self,
        config: DictConfig
    ):
        ###############################################################
        # 我自己加的变量！！！！！！！
        ###############################################################
        self.use_muzmark = config.muzmark.use
        self.use_wm_positioner = config.muzmark.wm_positioner.use
        self.LDPC_use = config.LDPC.use
        self.FSWE_use = config.muzmark.FSWE
        # delete
        print(f"{config.wm_model.wm_ratio=}")

        ###############################################################
        # Set up parameters for other function to use
        ###############################################################
        self.config = config
        self.test_path = config.dataset.test_path
        self.num_workers = config.get("num_workers", 4) # Set number of data loading workers (default: 4)
        self.msg_len = config.wm_model.message.len
        self.wm_ratio = config.wm_model.wm_ratio
        self.allow_missing_dataset = config.allow_missing_dataset

        ###############################################################
        # Set up device
        ###############################################################
        if torch.cuda.is_available() and str(config.device).startswith('cuda'):
            self.device = torch.device(f"cuda:{config.device_id}")
        elif torch.mps.is_available() and str(config.device).startswith('mps'):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # Check whether using device is different from request
        if str(self.device) != str(config.device):
            logger.info(f"Using device: {self.device} (requested: {config.device}).")

        ###############################################################
        # Set up sample rate & evaluation segment duration
        ###############################################################
        try:
            self.sample_rate = config.sample_rate
            self.eval_seg_duration = config.dataset.eval_seg_duration
        except AttributeError as e:
            raise AttributeError(f"Missing required config field: {e}.")
        self.num_samples  = int(self.eval_seg_duration * self.sample_rate)

        ###############################################################
        # Set up stft written in raw bench
        ###############################################################
        # Optional STFT setup
        stft_cfg = config.get("stft", None)
        if stft_cfg is not None:
            self.stft = STFT(
                filter_length=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_len,
                win_length=stft_cfg.win_len,
                num_samples=self.num_samples
            ).to(self.device)
        else:
            self.stft = None

        ###############################################################
        # Set up tools for further use
        ###############################################################
        self.music_augmentation = MusicAttack(
            sr=config.sample_rate,
            datapath=config.datapath,
            stft=self.stft,
            config=config.attack,
            ffmpeg4codecs=shutil.which("ffmpeg"),
            device=self.device.type
        )
        self.quality_evaluator = MusicEvaluator(config.sample_rate, device=self.device)

        ###############################################################
        # Run defined function below for initialization
        ###############################################################
        self.build_dataloaders()
        self.build_models(
            vocoder_checkpoint=config.wm_model.vocoder.vocoder_checkpoint,
            vocoder_config=config.wm_model.vocoder.vocoder_config
        )
        self.load_models(config.wm_model.model.test.model_path)

        ###############################################################
        # Set up LDPC is required to use
        ###############################################################
        if self.LDPC_use:
            self.ldpc = LDPC(
                num_codes=self.msg_len,
                row_weight=config.LDPC.row_weight,
                col_weight=config.LDPC.col_weight,
                device=self.device
            )
            self.real_msg_len = self.ldpc.get_real_msg_len()
        else:
            self.ldpc = None
            self.real_msg_len = 11


    def build_dataloaders(self):
        # validity of the data detail csv file
        if self.test_path is not None:
            assert os.path.isfile(self.test_path), f"Test path {self.test_path} is not a valid file."
        # get data details
        data = MusicDataset(
            dataset_filepath=self.test_path,
            datapath=self.config.datapath,
            config=self.config.dataset,
            num_samples=self.num_samples,
            mode="test",
            allow_missing_dataset=self.allow_missing_dataset
        )
        # set up torch dataloader
        self.dataloader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=init_worker
        )

    def build_models(
        self,
        vocoder_checkpoint: Union[Path, str],
        vocoder_config: Union[Path, str]
    ):
        """
        Build and initialize the encoder and decoder models.
        Args:
            vocoder_checkpoint: str
                Path to the pretrained vocoder checkpoint file.
            vocoder_config: str
                Path to configuration file for the vocoder.
        """
        if self.config.wm_model.model_type == "timbre":
            if self.use_muzmark:
                self.encoder = muzmark.MuzEncoder(
                    config=self.config,
                    device=self.device
                ).to(self.device)
            else:
                self.encoder = timbre.Encoder(
                    process_config=self.config.wm_model.process,
                    model_config=self.config.wm_model.model,
                    msg_length=self.msg_len,
                    device=self.device
                ).to(self.device)
            if self.use_muzmark:
                self.decoder = muzmark.MuzDecoder(
                    config=self.config,
                    device=self.device
                ).to(self.device)
            else:
                self.decoder = timbre.Decoder(
                    process_config=self.config.wm_model.process,
                    model_config=self.config.wm_model.model,
                    msg_length=self.msg_len,
                    vocoder_config=vocoder_config,
                    vocoder_checkpoint=vocoder_checkpoint,
                    device=self.device
                ).to(self.device)
        else:
            raise ValueError(f'Invalid model type {self.config.model_type}, expected "timbre", "wavmark" and "audioseal".')

    def load_models(
        self,
        checkpoint: Union[Path, str]
    ):
        """
        Load encoder and decoder model weights from a checkpoint.

        Args:
            checkpoint: Union[Path, str]
                Path to the checkpoint directory or file.
        """
        if not self.use_muzmark:
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
            logger.info(f"model <<{checkpoint}>> loaded.")

            self.encoder.load_state_dict(ckpt["encoder"])
            self.decoder.load_state_dict(ckpt["decoder"], strict=False)

    def transfer_to_model_sr(
        self,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Transfer the music data to the sample rate that the model required.

        Args:
            y: torch.Tensor
                The loaded music data in torch tensor.

        Returns:
            torch.Tensor
                Transfered music data.
        """
        if self.config.wm_model.model_type == "timbre":
            if self.config.sample_rate != 22050:
                return torchaudio.transforms.Resample(
                    orig_freq=self.config.sample_rate, new_freq=22050
                ).to(self.device)(y)
            else:
                return y.clone()
        elif self.config.model_type in ["wavmark", "audioseal"]:
            if self.config.sample_rate != 16000:
                return torchaudio.transforms.Resample(
                    orig_freq=self.config.sample_rate, new_freq=16000
                ).to(self.device)(y)
            else:
                return y.clone()

    def transfer_back_music_sr(
            self,
            y_wm: torch.Tensor
    ) -> torch.Tensor:
        """
        Transfer the watermarked music data back to the sample rate that the music data originally is.

        Args:
            y_wm: torch.Tensor
                The watermarked music data in torch tensor.

        Returns:
            Transfered watermarked music data.
        """
        if self.config.wm_model.model_type == "timbre":
            if self.config.sample_rate != 22050:
                return torchaudio.transforms.Resample(
                    orig_freq=22050, new_freq=self.config.sample_rate
                ).to(self.device)(y_wm)
            else:
                return y_wm.clone()
        elif self.config.model_type in ["wavmark", "audioseal"]:
            if self.config.sample_rate != 16000:
                return torchaudio.transforms.Resample(
                    orig_freq=16000, new_freq=self.config.sample_rate
                ).to(self.device)(y_wm)
            else:
                return y_wm.clone()

    def LDPC_encode(
        self,
        message: torch.Tensor
    ):
        msg = message[..., :self.real_msg_len]
        msg_bin = self.ldpc.pm1_bin_convert(msg, target="bin")
        codes = self.ldpc.encode(msg_bin)
        codes_pm1 = self.ldpc.pm1_bin_convert(codes, target="pm1").to(torch.float32)
        return codes_pm1

    def LDPC_decode(
        self,
        msg: torch.Tensor
    ):
        received_codes_pm1 = torch.sign(msg)
        received_codes = self.ldpc.pm1_bin_convert(received_codes_pm1, target="bin")
        decoded_msg_bin, corrected_codes, valid = self.ldpc.decode(received_codes)
        if valid:
            return self.ldpc.pm1_bin_convert(decoded_msg_bin, target="pm1")
        else:
            return msg[..., :self.real_msg_len]

    def evaluate(
        self,
        save: bool = False
    ):


        # Initialize recorder
        benchmark_results_recorder = list()

        with torch.inference_mode():
            # num = 0
            # for data_detail_row in qqdm(self.dataloader):
            for data_detail_row in self.dataloader:
                # if num < 206:
                #     num += 1
                #     continue
                # elif num == 206:
                #     num += 1
                # else:
                #     quit()
                #####################
                # add watermark
                #####################
                # load data and create message for watermarking
                music_chunk, music_filepath, dataset, att_types, att_params, chunk_indice, start_time = data_detail_row
                message = self.random_message(self.msg_len, music_chunk.shape[0])
                message = message.to(self.device).to(torch.float32)
                message = message.unsqueeze(1)  # unsqueeze to match the size with music chunk
                if self.LDPC_use:
                    message = self.LDPC_encode(message)
                # prepare start bits for wm block verification
                start_bits = message[..., :3] if self.use_wm_positioner else None
                # move the music to device
                y = music_chunk.to(self.device)

                # add watermark to the audio file
                if self.use_muzmark:
                    y_wm, wm_position_dict_encode = self.encoder(y, message, ratio=self.wm_ratio)
                    y_wm = y_wm[..., :y.size(2)]
                else:
                    # transfer data to sample rate that model required
                    y_model = self.transfer_to_model_sr(y)
                    # add watermark to music
                    y_wm_model, _, _, _ = self.encoder.test_forward(y_model, message, ratio=self.wm_ratio)
                    # transfer data back to sample rate that the music data originally is
                    y_wm = self.transfer_back_music_sr(y_wm_model)[..., :y.size(2)]

                #####################
                # detect watermark from music (not augmented)
                #####################
                
                if self.use_muzmark:
                    y_decoded, wm_position_dict_decode, y_pre_acc = self.decoder(y, start_bits) # detect watermark for original music
                    y_wm_decoded, wm_position_dict_decode_wm, y_wm_pre_acc = self.decoder(y_wm, start_bits) # detect watermark for watermarked music
                else:
                    y_decoded = self.decoder.test_forward(self.transfer_to_model_sr(y)) # detect watermark for original music
                    y_wm_decoded = self.decoder.test_forward(y_wm_model) # detect watermark for watermarked music
                
                # adjust decoded message if LDPC applied
                if self.LDPC_use:
                    y_decoded = self.LDPC_decode(y_decoded)
                    y_wm_decoded = self.LDPC_decode(y_wm_decoded)
                else:
                    y_decoded = y_decoded[..., :self.real_msg_len]
                    y_wm_decoded = y_wm_decoded[..., :self.real_msg_len]

                #####################
                # music augmentation
                #####################
                y_wm_augmented = torch.zeros_like(y_wm)
                y_augmented = torch.zeros_like(y)
                for b in range(y.size(0)):

                    # set up augmentation parameters
                    args = {} if att_params[b] is None else json.loads(att_params[b])
                    if att_types[b] == 'phase_shift':
                        # During test and validation, the phase shift parameter is in seconds.
                        args['shift'] = int(args['shift'] * self.sample_rate)
                    
                    # do music augmentation
                    y_wm_augmented[b, ...] = self.music_augmentation(
                        y_wm[b, ...], attack_type=att_types[b], **args
                    )
                    y_augmented[b, ...] = self.music_augmentation(
                        y[b, ...], attack_type=att_types[b], **args
                    )
                    
                    #####################
                    # detect watermark from augmented music
                    #####################
                    if self.use_muzmark:
                        y_aug_decoded, wm_position_dict_decode_aug, y_aug_pre_acc = self.decoder(
                            y_augmented[b:(b + 1), ...], start_bits
                        ) # detect watermark for original music
                        y_wm_aug_decoded, wm_position_dict_decode_wm_aug, y_wm_aug_pre_acc = self.decoder(
                            y_wm_augmented[b:(b + 1), ...], start_bits
                        ) # detect watermark for watermarked music
                    else:
                        y_aug_decoded = self.decoder.test_forward(
                            self.transfer_to_model_sr(y_augmented[b::(b + 1), ...])
                        ) # detect watermark for original music
                        y_wm_aug_decoded = self.decoder.test_forward(
                            self.transfer_to_model_sr(y_wm_augmented[b::(b + 1), ...])
                        ) # detect watermark for watermarked music

                    # adjust decoded message if LDPC applied
                    if self.LDPC_use:
                        y_aug_decoded = self.LDPC_decode(y_aug_decoded)
                        y_wm_aug_decoded = self.LDPC_decode(y_wm_aug_decoded)
                    else:
                        y_aug_decoded = y_aug_decoded[..., :self.real_msg_len]
                        y_wm_aug_decoded = y_wm_aug_decoded[..., :self.real_msg_len]



                    #####################
                    # record evaluation results
                    #####################

                    # adjust evaluation length for balanced comparison
                    message = message[..., 3:self.real_msg_len]

                    y_decoded = y_decoded[..., 3:]
                    y_wm_decoded = y_wm_decoded[..., 3:]
                    y_aug_decoded = y_aug_decoded[..., 3:]
                    y_wm_aug_decoded = y_wm_aug_decoded[..., 3:]

                    # initialize a recorder to save evaluation results
                    single_eval = dict()
                    single_eval["music_filepath"] = music_filepath[b]

                    # work on non-augmented audio
                    single_eval['NA_NWM_ACC'] = 1 if (y_decoded >= 0).eq(message >= 0).all() else 0
                    # calculate watermarked music quality
                    quality_eval = self.quality_evaluator(y[b, ...], y_wm[b, ...])
                    single_eval["NA_SISNR"] = quality_eval["SISNR"]
                    single_eval["NA_FAD"] = quality_eval["FAD"]
                    single_eval["NA_MCD"] = quality_eval["MCD"]

                    # calculate watermark accuracy for music (not augmented)

                    bitwise_acc = self.__bitwise_acc(
                        decoded=y_wm_decoded, message=message, msg_len=(self.real_msg_len - 3)
                    )
                    single_eval["NUM_BITS"] = bitwise_acc[1]
                    single_eval["NA_BW_WM_ACC"] = float(bitwise_acc[0])
                    single_eval["NA_90_WM_ACC"] = 1 if float(bitwise_acc[0]) >= 0.9 else 0
                    single_eval["NA_OA_WM_ACC"] = 1 if (y_wm_decoded >= 0).eq(message >= 0).all() else 0

                    # work on augmented audio
                    single_eval['A_NWM_ACC'] = 1 if (y_aug_decoded >= 0).eq(message >= 0).all() else 0
                    # calculate watermarked music quality after music augmentation
                    quality_eval_aug = self.quality_evaluator(y[b, ...], y_wm_augmented[b, ...])
                    single_eval["A_SISNR"] = quality_eval_aug["SISNR"]
                    single_eval["A_FAD"] = quality_eval_aug["FAD"]
                    single_eval["A_MCD"] = quality_eval_aug["MCD"]

                    # calculate watermark accuracy for augmented music
                    aug_bitwise_acc = self.__bitwise_acc(
                        decoded=y_wm_aug_decoded, message=message, msg_len=(self.real_msg_len - 3)
                    )
                    single_eval["A_BW_WM_ACC"] = float(aug_bitwise_acc[0])
                    single_eval["A_90_WM_ACC"] = 1 if float(aug_bitwise_acc[0]) >= 0.9 else 0
                    single_eval["A_OA_WM_ACC"] = 1 if (y_wm_aug_decoded >= 0).eq(message >= 0).all() else 0

                    # record accuracy of the watermark positioner if use watermark positioner
                    if self.use_muzmark and self.use_wm_positioner:
                        # record preamble accuracy
                        single_eval["NA_NWM_PRE_ACC"] = y_pre_acc["overall"][b]
                        single_eval["NA_WM_PRE_ACC"] = y_wm_pre_acc["overall"][b]
                        single_eval["A_NWM_PRE_ACC"] = y_aug_pre_acc["overall"][0]
                        single_eval["A_WM_PRE_ACC"] = y_wm_aug_pre_acc["overall"][0]

                        # not in use so far
                        wm_position_info_encode = wm_position_dict_encode[b]
                        wm_position_info_decode = wm_position_dict_decode[b]
                        wm_position_info_decode_wm = wm_position_dict_decode_wm[b]
                        wm_position_info_decode_aug = wm_position_dict_decode_aug[0]
                        wm_position_info_decode_wm_aug = wm_position_dict_decode_wm_aug[0]
                        if att_types[b] == "time_stretch":
                            single_eval["NA_NWM_WM_POI_STAR_ACC"], single_eval["NA_NWM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode, attack_type=att_types[b], **args
                            )
                            single_eval["NA_WM_WM_POI_STAR_ACC"], single_eval["NA_WM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_wm, attack_type=att_types[b], **args
                            )
                            single_eval["A_NWM_WM_POI_STAR_ACC"], single_eval["A_NWM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_aug, attack_type=att_types[b], **args
                            )
                            single_eval["A_WM_WM_POI_STAR_ACC"], single_eval["A_WM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_wm_aug, attack_type=att_types[b], **args
                            )
                        else:
                            single_eval["NA_NWM_WM_POI_STAR_ACC"], single_eval["NA_NWM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode
                            )
                            single_eval["NA_WM_WM_POI_STAR_ACC"], single_eval["NA_WM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_wm
                            )
                            single_eval["A_NWM_WM_POI_STAR_ACC"], single_eval["A_NWM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_aug
                            )
                            single_eval["A_WM_WM_POI_STAR_ACC"], single_eval["A_WM_WM_POI_FULL_ACC"] = check_wm_position_acc(
                                wm_position_info_encode, wm_position_info_decode_wm_aug
                            )









                    benchmark_results_recorder.append(single_eval)





        # if self.use_muzmark:
        #     if self.config.muzmark.wm_positioner.use:
        #         if self.config.muzmark.FSWE:
        #             os.makedirs("./outputs/muzmark_pos_no_fswe", exist_ok=True)
        #             pd.DataFrame(benchmark_results_recorder).to_csv(
        #                 f"./outputs/muzmark_pos_no_fswe/muzmark_pos_no_fswe_{'do-LDPC' if self.LDPC_use else 'no-LDPC'}_{self.wm_ratio}_results.csv",
        #                 index=False
        #             )
        #         else:
        #             os.makedirs("./outputs/muzmark_pos", exist_ok=True)
        #             pd.DataFrame(benchmark_results_recorder).to_csv(
        #                 f"./outputs/muzmark_pos/muzmark_pos_{'do-LDPC' if self.LDPC_use else 'no-LDPC'}_{self.wm_ratio}_results.csv",
        #                 index=False
        #             )
        #     else:
        #         os.makedirs("./outputs/muzmark", exist_ok=True)
        #         pd.DataFrame(benchmark_results_recorder).to_csv(
        #             f"./outputs/muzmark/muzmark_{'do-LDPC' if self.LDPC_use else 'no-LDPC'}_{self.wm_ratio}_results.csv",
        #             index=False
        #         )
        # else:
        #     os.makedirs("./outputs/timbre", exist_ok=True)
        #     pd.DataFrame(benchmark_results_recorder).to_csv(
        #         f"./outputs/timbre/timbre_{'do-LDPC' if self.LDPC_use else 'no-LDPC'}_{self.wm_ratio}_results.csv",
        #         index=False
        #     )

        os.makedirs("./final_outputs", exist_ok=True)
        if self.use_muzmark and self.use_wm_positioner and self.FSWE_use and self.LDPC_use:
            os.makedirs("./final_outputs/muzmark", exist_ok=True)
            pd.DataFrame(benchmark_results_recorder).to_csv(
                f"./final_outputs/muzmark/muzmark_{self.wm_ratio}_results.csv", index=False
            )
        elif self.use_muzmark and self.use_wm_positioner and self.FSWE_use and not self.LDPC_use:
            os.makedirs("./final_outputs/muzmark-no-LDPC", exist_ok=True)
            pd.DataFrame(benchmark_results_recorder).to_csv(
                f"./final_outputs/muzmark-no-LDPC/muzmark-no-LDPC_{self.wm_ratio}_results.csv", index=False
            )
        elif self.use_muzmark and self.use_wm_positioner and not self.FSWE_use and self.LDPC_use:
            os.makedirs("./final_outputs/muzmark-no-FSWE", exist_ok=True)
            pd.DataFrame(benchmark_results_recorder).to_csv(
                f"./final_outputs/muzmark-no-FSWE/muzmark-no-FSWE_{self.wm_ratio}_results.csv", index=False
            )
        elif self.use_muzmark and not self.use_wm_positioner and self.FSWE_use and self.LDPC_use:
            os.makedirs("./final_outputs/muzmark-no-POSI", exist_ok=True)
            pd.DataFrame(benchmark_results_recorder).to_csv(
                f"./final_outputs/muzmark-no-POSI/muzmark-no-POSI_{self.wm_ratio}_results.csv", index=False
            )
        elif not self.use_muzmark:
            os.makedirs("./final_outputs/timbre", exist_ok=True)
            pd.DataFrame(benchmark_results_recorder).to_csv(
                f"./final_outputs/timbre/timbre_{self.wm_ratio}_results.csv", index=False
            )
        else:
            print("No result recorded.")

        print(f"WM-RATIO: {self.wm_ratio} - EXPERIMENT COMPLETED")






    @staticmethod
    def random_message(
            nbits: int,
            batch_size: int
    ) -> torch.Tensor:
        """
        Generate a random message as a 0/1 tensor.

        Args:
            nbits: int
                Number of bits in the message.
            batch_size: int
                Number of messages to generate.

        Returns:
            torch.Tensor: Random message tensor.
        """
        if nbits == 0:
            return torch.tensor([])

        return torch.randint(0, 2, (batch_size, nbits)) * 2 - 1

    def __bitwise_acc(
            self,
            decoded: torch.Tensor,
            message: torch.Tensor,
            msg_len: int = 30
    ) -> tuple[torch.Tensor, int]:
        """
        Compute bitwise accuracy between decoded and original message tensors.

        Args:
            decoded: torch.Tensor
                Decoded message tensor.
            message: torch.Tensor
                Original message tensor.

        Returns:
            torch.Tensor: Bitwise accuracy.
        """
        return (decoded >= 0).eq(message >= 0).sum().float() / msg_len, msg_len












if __name__ == "__main__":
    # Load base and override configs
    eval_cfg = OmegaConf.load('./third_party/raw_bench/configs/eval.yaml')
    model_cfg = OmegaConf.load('./third_party/raw_bench/configs/timbre/skeleton.yaml')
    together_cfg = OmegaConf.load("./third_party/raw_bench/configs/timbre/eval_skeleton.yaml")

    # Merge override into base (shallow merge)
    cfg = OmegaConf.merge(eval_cfg, model_cfg, together_cfg)

    cfg.attack = OmegaConf.load('./third_party/raw_bench/configs/attack/attack_loose.yaml')
    cfg.dataset = OmegaConf.load('./third_party/raw_bench/configs/dataset/eval_default.yaml')
    cfg.datapath = OmegaConf.load('./third_party/raw_bench/configs/datapath/datapath.yaml')

    cfg.vocoder_checkpoint = "./third_party/raw_bench/wm_ckpts/timbre/watermarking_model/hifigan/model/VCTK_V1/generator_v1"
    cfg.vocoder_config = "./third_party/raw_bench/wm_ckpts/timbre/watermarking_model/hifigan/config.json"
    cfg.checkpoint = "./ckpts/timbre_ckpt/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar"
    cfg.dataset.test_path = "./third_party/raw_bench/data/test_loose.csv"
    cfg.dataset.eval_seg_duration = 6
    cfg.allow_missing_dataset = True
    cfg.datapath.AIR = "./data/AIR/AIR_1_4"
    cfg.datapath.DEMAND = "./data/DEMAND"
    cfg.datapath.Bach10 = "./data/Bach10"
    cfg.datapath.Freischuetz = "./data/Freischuetz"
    cfg.model_type = "timbre" # change watermarking model type
    cfg.sample_rate = 48000
    cfg.message.len = 30
    # cfg.ffmpeg4codecs = "/scratch/local/ssd2/zuocheng/ffmpeg-7.0.2-amd64-static/ffmpeg"
    #####################
    # 专门为加了LDPC而更改的
    #####################
    cfg.LDPC = dict()
    cfg.LDPC.use = True
    cfg.LDPC.real_msg_len = 11
    cfg.LDPC.row_weight = 2
    cfg.LDPC.col_weight = 3
    ##################
    # my configuration
    ##################
    cfg.wm_ratio = 1.0

    cfg.is_muztimbre = False

    # 我自己加的
    cfg.device_id = 0
    print(f"我们读进去的configs是：\n{OmegaConf.to_yaml(cfg)}")

    evaluator = BenchmarkPipeline(config=cfg)
    evaluator.evaluate()


# nohup python3 muzmark/benchmark/pipeline.py > logs/timbre_bm_1.0.log 2>&1 &


