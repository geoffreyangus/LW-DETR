#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------


export PYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths; print(get_paths()['include'])")
export TORCH_CUDA_ARCH_LIST="8.9"
python setup.py build install --user