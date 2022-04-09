#!/bin/bash
num_gpus=$1

mpirun -np $num_gpus -N $num_gpus --hostfile ./hfile.txt \
    python executionGraphReplay.py --master-ip $(head -n 1 ./hfile.txt) \
    --backend nccl --device cuda --n 5 --w 1 \
    --exgr-path ./execution_graphs/test/simple_fw

mpirun -np $num_gpus -N $num_gpus --hostfile ./hfile.txt \
    python executionGraphReplay.py --master-ip $(head -n 1 ./hfile.txt) \
    --backend nccl --device cuda --n 5 --w 1 \
    --exgr-path ./execution_graphs/test/simple_fw_bw