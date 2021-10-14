#!/bin/bash
num_gpus=$1
for collective in all_to_all all_to_allv all_reduce all_gather all_gather_base reduce reduce_scatter;
do
    mpirun -np ${num_gpus} -N ${num_gpus} --hostfile ./hfile.txt ./comms.py --master-ip $(head -n 1 ./hfile.txt) --b 4 --e 2G --n 200 --w 20 \
        --f 2 --z 1 --collective $collective --backend nccl --device cuda --log INFO > ${collective}_${num_gpus}.txt
done
