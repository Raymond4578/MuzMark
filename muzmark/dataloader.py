import os, sys
# 把上一级目录（cwd）插入到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from third_party.raw_bench.raw_bench.dataloader import AudioDataset

class MusicDataset(AudioDataset):
    def __init__(
        self,
        dataset_filepath: str,
        datapath: dict,
        config: argparse.Namespace = None,
        num_samples: int = 16000,
        random_start_frame: bool = False,
        csv_delimiter: str = '|',
        mode: str = 'train',
        allow_missing_dataset: bool = False
    ):
        super(MusicDataset, self).__init__(
            dataset_filepath=dataset_filepath,
            datapath=datapath,
            config=config,
            num_samples=num_samples,
            random_start_frame=random_start_frame,
            csv_delimiter=csv_delimiter,
            mode=mode,
            allow_missing_dataset=allow_missing_dataset
        )










if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Load base and override configs
    eval_cfg = OmegaConf.load('./third_party/raw_bench/configs/eval.yaml')
    model_cfg = OmegaConf.load('./third_party/raw_bench/configs/silentcipher/skeleton.yaml')
    together_cfg = OmegaConf.load("./third_party/raw_bench/configs/timbre/eval_skeleton.yaml")

    # Merge override into base (shallow merge)
    cfg = OmegaConf.merge(eval_cfg, model_cfg, together_cfg)

    cfg.attack = OmegaConf.load('./third_party/raw_bench/configs/attack/attack_strict.yaml')
    cfg.dataset = OmegaConf.load('./third_party/raw_bench/configs/dataset/eval_default.yaml')
    cfg.datapath = OmegaConf.load('./third_party/raw_bench/configs/datapath/datapath.yaml')

    cfg.vocoder_checkpoint = "./third_party/raw_bench/wm_ckpts/timbre/watermarking_model/hifigan/model/VCTK_V1/generator_v1"
    cfg.vocoder_config = "./third_party/raw_bench/wm_ckpts/timbre/watermarking_model/hifigan/config.json"
    cfg.checkpoint = "./ckpts/timbre_ckpt/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar"
    cfg.dataset.test_path = "./third_party/raw_bench/data/test_strict.csv"
    cfg.dataset.eval_seg_duration = 6
    cfg.allow_missing_dataset = True
    cfg.datapath.Bach10 = "./data/Bach10"
    cfg.datapath.Freischuetz = "./data/Freischuetz"
    cfg.sample_rate = 48000

    print(f"我们读进去的configs是：\n{OmegaConf.to_yaml(cfg)}")
    print(int(6 * cfg.sample_rate))

    dataset = MusicDataset(
        dataset_filepath=cfg.dataset.test_path,
        datapath=cfg.datapath,
        config=cfg.dataset,
        num_samples=int(6 * cfg.sample_rate),
        mode="test",
        allow_missing_dataset=cfg.allow_missing_dataset
    )

    # for data in dataset:
    #     print(data)