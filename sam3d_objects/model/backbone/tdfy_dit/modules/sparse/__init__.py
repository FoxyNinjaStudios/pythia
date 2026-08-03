# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import *
from loguru import logger
import torch

# Reconstruction runs on the pure-PyTorch sparse path (naive gather/scatter),
# which executes on CPU. The upstream native-Metal reconstruction kernels were
# removed (see the README, "How the port works"); there is no GPU backend to
# select for reconstruction. SAM and MoGe keep their own MPS path elsewhere.
BACKEND = "mps"   # selects the conv_mps / MPSSparseConvTensor pure-PyTorch path
DEBUG = False
ATTN = "sdpa"     # torch.nn.functional.scaled_dot_product_attention


def __from_env():
    import os

    global DEBUG

    env_sparse_debug = os.environ.get("SPARSE_DEBUG")
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == "1"

    logger.info(f"[SPARSE] Backend: {BACKEND}, Attention: {ATTN}")


__from_env()


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .basic import *
from .norm import *
from .nonlinearity import *
from .linear import *
from .attention import *
from .conv import *
from .spatial import *
from . import transformer
