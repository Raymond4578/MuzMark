import numpy as np
import torch
import torch.nn as nn
import pyldpc
import ldpc.code_util
from ldpc import BpDecoder
from typing import Union



class LDPC(nn.Module):
    def __init__(
        self,
        num_codes: int = 30,
        row_weight: int = 2,
        col_weight: int = 3,
        device: Union[torch.device, str] = "cuda:0"
    ):
        super(LDPC, self).__init__()
        self.n = num_codes
        self.col_weight = col_weight
        self.row_weight = row_weight
        self.device = device

        # get LDPC parity-check matrix, generator matrix
        self.H, G = pyldpc.make_ldpc(self.n, self.row_weight, self.col_weight, systematic=True, sparse=True, seed=4)
        self.G = G.T
        self.H_tensor = torch.from_numpy(self.H).to(self.device)
        self.G_tensor = torch.from_numpy(self.G).to(self.device)
        self.m = self.H.shape[0]

        # get msg_len k
        _, self.k, self.d_estimate = ldpc.code_util.compute_code_parameters(self.H)

        # initialise BP Decoder
        self.bp_decoder = BpDecoder(
            pcm=self.H,
            error_rate=0.1,
            max_iter=self.n,
            bp_method="product_sum"
        )

    def get_real_msg_len(self):
        return self.G_tensor.size(0)

    def check_codes_validity(self, codes: torch.Tensor) -> torch.Tensor:
        temp = (self.H_tensor.float() @ codes.float().unsqueeze(3) % 2).squeeze(3).long()
        return (temp == 0).all(dim=-1)

    @staticmethod
    def pm1_bin_convert(code: torch.Tensor, target: str = "bin") -> torch.Tensor:
        if target == "pm1":
            return 1 - 2 * code
        elif target == "bin":
            return (1 - code) // 2

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        # check whether the length of msg is valid
        if msg.size(2) != self.k:
            raise ValueError(f"The input message length is {msg.size(2)}, but it should be {self.k}.")

        return (msg.unsqueeze(2).float() @ self.G_tensor.float() % 2.0).squeeze(2).long()

    def decode(self, received_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corrected_codes = torch.zeros_like(received_codes)
        for b in range(received_codes.size(0)):
            single_codes = received_codes[b, 0, ...].detach().to("cpu").numpy()
            corrected_codes[b, 0, :] = torch.from_numpy(self.bp_decoder.decode(single_codes))

        return corrected_codes[..., :self.k], corrected_codes, self.check_codes_validity(corrected_codes)




if __name__ == "__main__":
    ldpc = LDPC()
    msg = torch.randint(low=0, high=2, size=[10, 1, 11])

    if torch.cuda.is_available():
        msg = msg.to(device=f"cuda:0")

    codes = ldpc.encode(msg)

    codes_pm1 = ldpc.pm1_bin_convert(codes, target="pm1")

    # def random_swap(code, rate=0.1):
    #     error_ls = np.random.choice([0, 1], size=len(code), p=[1 - rate, rate])
    #     indices = np.where(error_ls == 1)[0]
    #     output_codes = code.copy()
    #     print(f"{indices}")
    #     for i in indices:
    #         output_codes[i] = int(1 - output_codes[i])
    #     return output_codes

    # print(f"{codes=}")
    # received_codes = random_swap(codes)
    received_codes_pm1 = codes_pm1
    # print(f"{received_codes=}")

    received_codes = ldpc.pm1_bin_convert(received_codes_pm1, target="bin")

    decoded_msg, corrected_codes, valid = ldpc.decode(received_codes)


    # print(
    #     f"The original message is {msg},", "\n"
    #     f"The decoded- message is {decoded_msg}.", "\n"
    #     "\n"
    #     f"The original- codes is {codes},", "\n"
    #     f"The received- codes is {received_codes},", "\n"
    #     f"The corrected codes is {corrected_codes}.", "\n"
    #     f"This corrected code is a valid codes: {valid}."
    # )



