#!/bin/bash
# BSD 3-Clause License
#
# Copyright (c) 2021, The Regents of the University of California, Davis
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

mkdir -p examples/pytorch/my_eg_jsons
python setup.py install

# AlexNet
rm examples/pytorch/my_eg_jsons/alex_net.json
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m alex_net -w 10 -i 100 # -v
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m alex_net -w 10 -i 100 -p
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m alex_net -w 10 -i 100 --subgraph "module::ZeroGrad"
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m alex_net -w 10 -i 100 --subgraph "module::Forward"
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m alex_net -w 10 -i 100 --subgraph "module::Backward_WeightsUpdate"

# DLRM default (batch_size = 1024)
# Ref time on V100: ~5.6 ms
# Bottom MLP: 512-512-64
# Top MLP: 1024-1024-1024-1
# Embedding lookup: 1000000 rows * 8 tables, D = 64, L = 100
python -m param_bench.train.compute.python.pytorch.my_eg_replay -m dlrm_default_1024 -w 10 -i 100 -k "module::get_batch_data"
