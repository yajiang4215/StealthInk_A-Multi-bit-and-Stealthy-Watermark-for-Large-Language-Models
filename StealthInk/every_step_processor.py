# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations  
import numpy as np
from scipy.stats import t
from scipy.stats import binom 
import collections
from math import sqrt, ceil  
import time
import scipy.stats
import numpy as np 
import torch 
from torch import Tensor
from tokenizers import Tokenizer
import math
from normalizers import normalization_strategy_lookup
from hash_scheme import prf_lookup, seeding_scheme_lookup

class WatermarkBase:
    def __init__(
        self,
        vocab = None,
        seeding_scheme = "simple_3",  # mostly unused/always default
        hash_key = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
    ):

        # watermarking parameters
        self.vocab = vocab
        self.rng = None
        self.vocab_size = len(vocab)
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        # print("seeding scheme:", self.seeding_scheme)
        self.rng = torch.Generator(device=input_ids.device)
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme
        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)
            prf_key = prf_lookup[self.prf_type](input_ids[-1:], salt_key=self.hash_key)
            # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
            self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long
        elif seeding_scheme == "simple_3":
            assert input_ids.shape[-1] >= 3, f"seeding_scheme={seeding_scheme} requires at least a 3 token prefix sequence to seed rng"
            self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)
            
            prf_key = prf_lookup[self.prf_type](input_ids[:, -3:], salt_key=self.hash_key)
            # print("input_ids[-3:]:", input_ids[:, -3:], prf_key)
            self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long
            # torch.manual_seed(prf_key % (2**64 - 1))
        return



class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = 'cuda:0',
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        elif self.seeding_scheme == "simple_3":
            self.min_prefix_len = 3

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_p_value(self, z):
        p_value = 1-scipy.stats.norm.sf(z)
        return p_value


    def hamming_distance(self, bit_str1, bit_str2):
        if len(bit_str1) != len(bit_str2):
            raise ValueError("The input strings must have the same length.")
        distance = 0
        for char1, char2 in zip(bit_str1, bit_str2):
            if char1 != char2:
                distance += 1

        return distance
