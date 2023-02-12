#!/bin/bash
num_gpus=$1
mkdir -p bench_results
# Benchmark all collectives
for collective in all_to_all all_to_allv all_reduce all_gather all_gather_base reduce reduce_scatter;
do
    mpirun -np ${num_gpus} --hostfile ./hfile.txt ./comms.py \
        --master-ip $(head -n 1 ./hfile.txt) --b 4 --e 2G --n 200 --w 20 --f 2 --z 1 --log INFO \
        --collective $collective --backend nccl --device cuda > ./bench_results/${collective}_${num_gpus}.txt
done

# Benchmark all_reduce with arbitrary size for testing
mpirun -np ${num_gpus} --hostfile ./hfile.txt ./comms.py \
    --master-ip $(head -n 1 ./hfile.txt) --b 5 --e 2G --n 200 --w 20 --f 1.3 --z 1 --log INFO \
    --collective all_reduce --backend nccl --device cuda > ./bench_results/general_all_reduce_${num_gpus}.txt

# Benchmark all_to_all with params from a file
mpirun -np ${num_gpus} --hostfile ./hfile.txt ./comms.py \
    --master-ip $(head -n 1 ./hfile.txt) --n 100 --w 20 --z 1 --log INFO \
    --bench-params-file ${PM_HOME}/bench_params/a2a_${num_gpus}_params.txt \
    --collective "all_to_allv" --backend nccl --device cuda > ./bench_results/general_all_to_allv_${num_gpus}.txt
