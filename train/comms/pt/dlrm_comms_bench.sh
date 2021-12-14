#!/bin/bash
model_name=$1
num_gpus=$2

if [[ $model_name == "DLRM_vipul" ]]; # From Vipul
then
    _args=" --arch-mlp-bot=13-512-256-64-16\
            --arch-mlp-top=512-256-128-1\
            --arch-sparse-feature-size=16\
            --arch-embedding-size=1461-584-10131227-2202608-306-24-12518-634-4-93146-5684-8351593-3195-28-14993-5461306-11-5653-2173-4-7046547-18-16-286181-105-142572\
            --num-indices-per-lookup=38 "
elif [[ $model_name == "DLRM_default" ]]; # DLRM original
then
    _args=" --arch-mlp-bot=512-512-64\
            --arch-mlp-top=1024-1024-1024-1\
            --arch-sparse-feature-size=64\
            --arch-embedding-size=1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000\
            --num-indices-per-lookup=100\
            --num-indices-per-lookup-fixed "
elif [[ $model_name == "DLRM_DDP" ]]; # DLRM DDP example
then
    _args=" --arch-mlp-bot=128-128-128-128\
            --arch-mlp-top=512-512-512-256-1\
            --arch-sparse-feature-size=128\
            --arch-embedding-size=80000-80000-80000-80000-80000-80000-80000-80000"
fi

mpirun -np 4 -N 4 --hostfile ./hfile.txt ./dlrm.py --master-ip $(head -n 1 ./hfile.txt) --mini-batch-size 2048     --num-batches 100     --arch-mlp-bot=128-128-128-128    --arch-mlp-top=512-512-512–256-1    --arch-sparse-feature-size 128     --arch-embedding-size "80000-80000-80000-80000-80000-80000-80000-80000"
