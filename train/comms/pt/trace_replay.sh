#!/bin/bash
num_gpus=$1

if [[ $num_gpus == "1" ]];
then
    python commsTraceReplay.py --master-ip $(head -n 1 ./hfile.txt) --backend nccl --device cuda --trace-path ./traces/DLRM_MLPerf/1_2048
else
    mpirun -np $num_gpus -N $num_gpus --hostfile ./hfile.txt python commsTraceReplay.py --master-ip $(head -n 1 ./hfile.txt) --backend nccl --device cuda --trace-path ./traces/DLRM_MLPerf/1_2048
fi