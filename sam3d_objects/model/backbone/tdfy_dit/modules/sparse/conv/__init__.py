# Copyright (c) Meta Platforms, Inc. and affiliates.
from .. import BACKEND


SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'


def __from_env():
    import os

    global SPCONV_ALGO
    env_spconv_algo = os.environ.get("SPCONV_ALGO")
    if env_spconv_algo is not None and env_spconv_algo in [
        "auto",
        "implicit_gemm",
        "native",
    ]:
        SPCONV_ALGO = env_spconv_algo
    if BACKEND != "mps":
        print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")


__from_env()

if BACKEND == "torchsparse":
    from .conv_torchsparse import *
elif BACKEND == "spconv":
    from .conv_spconv import *
elif BACKEND == "metal":
    # Native Metal kernels with PyObjC (fastest, requires pyobjc-framework-Metal)
    try:
        from .conv_metal import MetalSparseConv3d as SparseConv3d
        from .conv_metal import MetalSubMConv3d as SubMConv3d
        from .conv_mps import SparseInverseConv3d  # Fall back for inverse conv
        print("[SPARSE][CONV] Using Metal compute shaders (PyObjC)")
    except ImportError as e:
        print(f"[SPARSE][CONV] Metal unavailable ({e}), falling back to MPS")
        from .conv_mps import SparseConv3d, SparseInverseConv3d
        from .conv_mps import MPSSubMConv3d as SubMConv3d
elif BACKEND == "mps":
    # PyTorch MPS with gather/scatter (slower, no external deps)
    from .conv_mps import SparseConv3d, SparseInverseConv3d
    from .conv_mps import MPSSubMConv3d as SubMConv3d
    print("[SPARSE][CONV] Using MPS (Apple Metal) backend")

