#!/bin/bash
num_gpus=$1

mpirun -np $num_gpus -N $num_gpus --hostfile ./hfile.txt \
    python executionGraphReplay.py --master-ip $(head -n 1 ./hfile.txt) \
    --backend nccl --device cuda --n 100 --w 20 \
    --exgr-path ./execution_graphs/DLRM_MLPerf/2_2048
