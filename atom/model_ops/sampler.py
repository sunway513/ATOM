# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter import mixed_sample_outer_exponential
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-10

    def forward(
        self,
        logits: torch.Tensor,  # (token_num, vocab_size)
        temperatures: torch.Tensor,  # (token_num,)
    ) -> torch.Tensor:  # (token_num,)
        sampled_tokens = torch.empty(
            logits.size(0), dtype=torch.int, device=logits.device
        )
        exponential = (
            torch.empty((1, logits.shape[-1]), dtype=torch.float, device=logits.device)
            .exponential_(1)
            .expand(*logits.shape)
        )
        mixed_sample_outer_exponential(
            sampled_tokens, logits, exponential, temperatures, eps=self.eps
        )
        return sampled_tokens
