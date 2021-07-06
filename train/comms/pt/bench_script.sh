#!/bin/bash
for collective in all_to_all all_to_allv all_reduce all_gather all_gather_base reduce reduce_scatter;
do
    mpirun -np 2 -N 2 --hostfile ./hfile.txt ./comms.py --master-ip $(head -n 1 ./hfile.txt) --b 8 --e 256M --n 100 \
        --f 2 --z 1 --collective $collective --backend nccl --device cuda --log INFO > ${collective}.txt
done
