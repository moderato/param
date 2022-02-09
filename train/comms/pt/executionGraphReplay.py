#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json
import logging
import time
from os import path

import torch
import comms_utils
import numpy as np
from comms_utils import paramCommsBench, paramTimer, paramProfile
from exec_graph_utils import ExecutionGraph, NodeType
from kernel_benchmark import *

logger = logging.getLogger(__name__)

def is_op(node):
    return (node.type == NodeType.OPERATOR and node.parent.type != NodeType.OPERATOR)


def is_forward(node):
    forward = False
    tmp = node
    while tmp.parent is not None:
        if 'DLRM forward' in tmp.name:
            return True
        tmp = tmp.parent

    return forward


CONSIDER = ["aten::linear", "AddmmBackward", "aten::bmm", "BmmBackward0", "aten::matmul", "MmBackward", \
                "aten::conv2d", "CudnnConvolutionBackward", \
                "LookupFunction", "LookupFunctionBackward", \
                "aten::batch_norm", "CudnnBatchNormBackward", \
                "aten::index", "IndexBackward", \
                "aten::relu", "aten::relu_", "ReluBackward0", "ReluBackward1", \
                "aten::sigmoid", "SigmoidBackward", \
                "aten::binary_cross_entropy", "BinaryCrossEntropyBackward", \
                "aten::mse_loss", "MseLossBackward", \
                "aten::avg_pool2d", "AvgPool2D", \
                "aten::max_pool2d", "MaxPool2DWithIndicesBackward", \
                "aten::add", "aten::add_", "aten::__and__", "aten::cat", "aten::sum", "aten::to", "aten::ones_like", \
                "torch::autograd::AccumulateGrad", "Optimizer.step#SGD.step", "Optimizer.zero_grad#SGD.zero_grad"]

SKIP = ["aten::ones", "SliceBackward", "FusedDropoutBackward"]


COMMS = ["nccl:all_to_all", "nccl:all_reduce"]


def run_op(op, op_lists, iters, warmup_iters, global_rank):
    t = 0
    if op.name == "aten::linear":
        transpose = None
        for child in op.children:
            if child.name == "aten::transpose":
                transpose = child
            elif child.name == "aten::t":
                transpose = child.children[0]
            elif "addmm" in child.name:
                if transpose is not None:
                    M, N = transpose.input_shapes[0][0], transpose.input_shapes[0][1]
                    if (transpose.inputs[0], transpose.inputs[1]) == (1, 2):
                        trans_type = 0
                    elif (transpose.inputs[0], transpose.inputs[1]) == (0, 2):
                        trans_type = 1
                    else: # (0, 1)
                        trans_type = 2
                    t += benchmark_transpose(1, M, N, trans_type, iters, warmup_iters)
                op_lists["addmm"].append(child)
                M, K, N = child.input_shapes[1][0], child.input_shapes[1][1], child.input_shapes[2][1]
                t = benchmark_linear(M, N, K, iters, warmup_iters)
            elif child.name == "aten::matmul":
                op_lists["mm"].append(child)
                M, K, N = child.input_shapes[0][0] * child.input_shapes[0][1], child.input_shapes[0][2], child.input_shapes[1][1] if len(child.input_shapes[1]) > 1 else 1
                t = benchmark_linear(M, N, K, iters, warmup_iters)
    elif op.name == "AddmmBackward":
        addmm_op = op_lists["addmm"].pop()
        M, K, N = addmm_op.input_shapes[1][0], addmm_op.input_shapes[1][1], addmm_op.input_shapes[2][1]
        t = benchmark_linear(M, N, K, iters, warmup_iters, backward=True)
    elif op.name == "MmBackward":
        mm_op = op_lists["mm"].pop()
        M, K, N = mm_op.input_shapes[0][0] * mm_op.input_shapes[0][1], mm_op.input_shapes[0][2], mm_op.input_shapes[1][1] if len(mm_op.input_shapes[1]) > 1 else 1
        t = benchmark_linear(M, N, K, iters, warmup_iters, backward=True)
    elif op.name == "aten::matmul":
        for child in op.children:
            if "aten::bmm" in child.name:
                op_lists["bmm"].append(child)
                batch_size, M, K, N = child.input_shapes[0][0], child.input_shapes[0][1], child.input_shapes[0][2], child.input_shapes[1][2]
                t = benchmark_fc(batch_size, M, N, K, iters, warmup_iters)
    elif op.name == "aten::bmm":
        op_lists["bmm"].append(op)
        batch_size, M, K, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2], op.input_shapes[1][2]
        t = benchmark_fc(batch_size, M, N, K, iters, warmup_iters)
    elif op.name == "BmmBackward0":
        bmm_op = op_lists["bmm"].pop()
        batch_size, M, K, N = bmm_op.input_shapes[0][0], bmm_op.input_shapes[0][1], bmm_op.input_shapes[0][2], bmm_op.input_shapes[1][2]
        t = benchmark_fc(batch_size, M, N, K, iters, warmup_iters, backward=True)
    elif op.name == "aten::conv2d":
        op_lists["conv2d"].append(op)
        batch_size, IC, IH, IW = op.input_shapes[0]
        OC, FH, FW = op.input_shapes[1][0], op.input_shapes[1][2], op.input_shapes[1][3]
        stride, _, dilation, is_dw = op.inputs[3][0], op.inputs[4][0], op.inputs[5][0], int(op.inputs[6] != 1)
        t = benchmark_conv2d(batch_size, IH, IW, IC, OC, stride, dilation, FH, FW, is_dw, iters, warmup_iters)
    elif op.name == "CudnnConvolutionBackward":
        conv_op = op_lists["conv2d"].pop()
        batch_size, IC, IH, IW = conv_op.input_shapes[0]
        OC, FH, FW = conv_op.input_shapes[1][0], conv_op.input_shapes[1][2], conv_op.input_shapes[1][3]
        stride, _, dilation, is_dw = conv_op.inputs[3][0], conv_op.inputs[4][0], conv_op.inputs[5][0], int(conv_op.inputs[6] != 1)
        t = benchmark_conv2d(batch_size, IH, IW, IC, OC, stride, dilation, FH, FW, is_dw, iters, warmup_iters, backward=True)
    elif op.name == "LookupFunction":
        op_lists["el"].append(op)
        T = op.input_shapes[1][0]
        D = op.input_shapes[0][1]
        B = int((op.input_shapes[3][0] - 1) / T)
        E = int(op.input_shapes[0][0] / T)
        L = int(op.input_shapes[2][0] / B / T)
        rows_per_block = max(int(256 / D), 1)
        t = benchmark_embedding_lookup(B, E, T, L, D, rows_per_block, iters, warmup_iters, backward=False, shmem=True, sgd=True)
    elif op.name == "LookupFunctionBackward":
        el_op = op_lists["el"].pop()
        T = el_op.input_shapes[1][0]
        D = el_op.input_shapes[0][1]
        B = int((el_op.input_shapes[3][0] - 1) / T)
        E = int(el_op.input_shapes[0][0] / T)
        L = int(el_op.input_shapes[2][0] / B / T)
        rows_per_block = max(int(256 / D), 1)
        t = benchmark_embedding_lookup(B, E, T, L, D, rows_per_block, iters, warmup_iters, backward=True, shmem=True, sgd=True)
    elif op.name == "aten::batch_norm":
        op_lists["bn"].append(op)
        if len(op.input_shapes[0]) == 4:
            batch_size, OC, H, _ = op.input_shapes[0] # BN 2D
        elif len(op.input_shapes[0]) == 3:
            batch_size, OC, H = op.input_shapes[0] # BN 1D with 3D input
        else:
            batch_size, OC = op.input_shapes[0] # BN 1D with 2D input
            H = 1
        t = benchmark_bn(batch_size, H, H, OC, iters, warmup_iters)
    elif op.name == "CudnnBatchNormBackward":
        bn_op = op_lists["bn"].pop()
        batch_size, OC, H, _ = bn_op.input_shapes[0]
        t = benchmark_bn(batch_size, H, H, OC, iters, warmup_iters, backward=True)
    elif op.name == "aten::index":
        op_lists["tril"].append(op)
        batch_size, M, N = op.input_shapes[0][0], op.input_shapes[0][1], op.input_shapes[0][2]
        total_output_element = op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = benchmark_tril(batch_size, M, N, diag, iters, warmup_iters)
    elif op.name == "IndexBackward": # See all kernels as a whole
        tril_op = op_lists["tril"].pop()
        batch_size, M, N = tril_op.input_shapes[0][0], tril_op.input_shapes[0][1], tril_op.input_shapes[0][2]
        total_output_element = tril_op.input_shapes[1][1][0]
        if total_output_element == int(M * (1+N) / 2):
            diag = 1
        else:
            diag = 0
        t = benchmark_tril(batch_size, M, N, diag, iters, warmup_iters, backward=True)
    elif op.name == "aten::cat":
        sizes = [tuple(s) for s in op.input_shapes[0] if s]
        dim = op.inputs[-1]
        t = benchmark_concat(sizes, dim, iters, warmup_iters)
    elif op.name == "aten::to":
        device = "cuda:{}".format(global_rank)
        if device in op.inputs:
            s = np.prod(op.input_shapes[0])
            t = benchmark_memcpy(s, iters, warmup_iters)
    elif op.name in ["aten::relu", "aten::relu_"]:
        op_lists["relu"].append(op)
        s = np.prod(op.input_shapes[0])
        t = benchmark_relu(s, iters, warmup_iters)
    elif op.name in ["ReluBackward0", "ReluBackward1"]:
        relu_op = op_lists["relu"].pop()
        s = np.prod(relu_op.input_shapes[0])
        t = benchmark_relu(s, iters, warmup_iters, backward=True)
    elif op.name == "aten::max_pool2d" or op.name == "aten::max_pool2d_with_indices":
        op_lists["max_pool"].append(op)
        batch_size, C, H, W = op.input_shapes[0]
        FH, FW, stride, dilation = op.inputs[1][0], op.inputs[1][1], op.inputs[2][0], op.inputs[4][0]
        t = benchmark_pool(batch_size, H, W, C, stride, dilation, FH, FW, "max", iters, warmup_iters)
    elif op.name == "aten::avg_pool2d" or op.name == "aten::avg_pool2d_with_indices":
        op_lists["avg_pool"].append(op)
        batch_size, C, H, W = op.input_shapes[0]
        FH, FW, stride, dilation = op.inputs[1][0], op.inputs[1][1], op.inputs[2][0], op.inputs[4][0]
        t = benchmark_pool(batch_size, H, W, C, stride, dilation, FH, FW, "avg", iters, warmup_iters)
    # print(op.name, op.input_shapes, t)
    return t


class ExgrReplayBench(paramCommsBench):
    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-dist", "pytorch-xla-tpu"])
        self.exgr = {}
        self.exgr_file = ""
        self.is_dry_run = False
        self.is_blocking = True
        self.do_warm_up = True
        self.allowList = ""

        self.strToTorchDtype = {
            "Byte": torch.uint8,
            "Float": torch.float32,
            "Int": torch.int32,
            "Long": torch.long,
            "Double": torch.double,
        }

    def readArgs(self, parser):
        # read the common/basic arguments
        super().readArgs(parser)
        parser.add_argument(
            "--exgr-path",
            type=str,
            default="./",
            help="File path to read the execution graph. All rank read their own execution graph file unless `--use-one-exgr` is used.",
        )
        parser.add_argument(
            "--use-one-exgr",
            action="store_true",
            default=False,
            help="Toggle to use only one execution graph for all ranks",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=self.is_dry_run,
            help="Toggle to only analyze execution graph without actually replaying collectives",
        )
        parser.add_argument(
            "--no-warm-up",
            action="store_true",
            default=False,
            help="Toggle to disable performing extra replaying for warm-up",
        )
        parser.add_argument(
            "--allow-ops",
            "--allow-list",
            type=str,
            default="all",
            help="List of desired collectives (separate by comma) to be replayed, e.g., `--allow-ops all_reduce,all_to_allv,wait`, typo or not supported collectives will be ignored.",
        )
        parser.add_argument(
            "--n",
            type=int,
            default=5,
            help="Number of iterations"
        )  # number of iterations
        parser.add_argument(
            "--w",
            type=int,
            default=5,
            help="Number of warmup iterations"
        )  # number of warmup-iterations
        return parser.parse_args()

    def setExgrFile(self, args, mpi_env_params):
        self.exgr_file = (
            f"{args.exgr_path}/rank{mpi_env_params['global_rank']}.json"
        )

    def checkArgs(self, args):
        super().checkArgs(args)

        if (path.exists(self.exgr_file) is False
            or path.isfile(self.exgr_file) is False
        ):
            logger.error(
                f"Execution graphfile {self.exgr_file} not exist or not a file! Please specifiy the correct path using --exgr-path"
            )
            comms_utils.gracefulExit()

    def readGraph(self):
        """Read execution graph file from remote server or local disk"""
        # read the json file from local disk
        with open(self.exgr_file) as f:
            self.exgr = ExecutionGraph(json.load(f))

    def runComms(self, collName):
        collTimer = paramTimer()

        if self.is_blocking:
            self.backendFuncs.sync_barrier(self.collectiveArgs)
        # replay the collective
        with paramProfile(
            timer=collTimer, description="# PARAM replay: {}".format(collName)
        ):
            if collName in self.backendFuncs.collectiveFunc.keys():
                self.backendFuncs.collectiveFunc[collName](
                    self.collectiveArgs, retFlag=True
                )
            if self.is_blocking:
                self.backendFuncs.complete_accel_ops(self.collectiveArgs)

        if self.is_blocking:
            with paramProfile(
                description="# PARAM replay barrier #"
            ) as bt:
                self.backendFuncs.sync_barrier(self.collectiveArgs)

        latency = collTimer.getTimeUS()
        global_latency = latency + (bt.intervalNS / 1e3)

        return (latency, global_latency)

    def allGatherSizes(self, size):
        # Push the list to device, then do an all-gather.
        sizeTensor = torch.tensor(
            size, device=self.backendFuncs.get_device()
        )
        self.collectiveArgs.opTensor = None
        sizeList = [
            torch.ones_like(sizeTensor) for _ in range(self.comm_size)
        ]
        self.collectiveArgs.opTensor = sizeList

        self.collectiveArgs.ipTensor = sizeTensor
        self.collectiveArgs.asyncOp = False
        self.collectiveArgs.dataSize = (
            sizeTensor.nelement() * sizeTensor.element_size()
        )
        self.collectiveArgs.numElements = sizeTensor.nelement()

        # use allgather as all process group should support it
        self.backendFuncs.all_gather(self.collectiveArgs)
        self.backendFuncs.complete_accel_ops(self.collectiveArgs)

        return sizeList

    def prepComms(self, op):
        size = op.input_shapes[0][0]
        all_sizes = self.allGatherSizes(size)
        # allocate tensors
        self.collectiveArgs.ipTensor = self.backendFuncs.alloc_random(
            [size],
            curRankDevice=self.collectiveArgs.device,
            dtype=self.strToTorchDtype["Float"], # TODO: more options
        )
        if op.name in ["nccl:all_to_all"]:
            # alltoall requires two tensors
            self.collectiveArgs.opTensor = self.backendFuncs.alloc_random(
                [sum([s.item() for s in all_sizes])],
                curRankDevice=self.collectiveArgs.device,
                dtype=self.strToTorchDtype["Float"], # TODO: more options
            )
        elif op.name in ["nccl:all_gather"]:
            # allgather requires a tensor list, e.g., List[torch.Tensor]
            self.collectiveArgs.opTensor = []
            for _ in range(self.collectiveArgs.world_size):
                self.collectiveArgs.opTensor.append(
                    self.backendFuncs.alloc_empty(
                        [sum([s.item() for s in all_sizes])],
                        curRankDevice=self.collectiveArgs.device,
                        dtype=self.strToTorchDtype["Float"], # TODO: more options
                    )
                )
        else:
            # only one tensor required for allreduce, reduce and broadcast
            self.collectiveArgs.opTensor = self.collectiveArgs.ipTensor

    def benchTime(self):
        op_lists = {
            "addmm": [],
            "bce": [],
            "bmm": [],
            "bn": [],
            "conv2d": [],
            "el": [],
            "mm": [],
            "mse": [],
            "relu": [],
            "sigmoid": [],
            "tril": [],
            "max_pool": [],
            "avg_pool": []
        }

        nodes = self.exgr.get_nodes(clean=True)
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
        sorted_nodes = [(id, node) for id, node in sorted_nodes if is_op(node)] # Filter modules, kernels, etc

        time_per_iter = 0
        for _, node in sorted_nodes:
            if node.name in SKIP:
                continue
            if node.name in "record_param_comms":
                op = None
                # Get the collective node
                def dfs(n):
                    nonlocal op
                    if n.name in COMMS:
                        op = n
                    for c in n.children:
                        dfs(c)
                dfs(node)
                if op is not None:
                    self.prepComms(op)
                    latency, _ = self.runComms(op.name.split("nccl:")[-1])
                    time_per_iter += latency / 1e3 # In ms
            if node.name in CONSIDER:
                # TODO: use PARAM's compute microbenchmark for this
                t = 1e3 * run_op(node, op_lists, self.numIters, self.numWarmupIters, self.global_rank) # In ms
                time_per_iter += t
        print("Rank {}: {:.2f} ms".format(self.global_rank, time_per_iter))

    def reportBenchTime(self, *args, **kwargs):
        pass

    def initBench(self, args):
        self.is_dry_run = args.dry_run
        self.is_blocking = args.z
        self.do_warm_up = not args.no_warm_up
        self.allowList = args.allow_ops
        self.numWarmupIters = args.w
        self.numIters = args.n

    def setBench(self, comms_world_info, commsParams):
        # init backend and corresponding function pointers
        if commsParams.nw_stack == "pytorch-dist":
            from pytorch_dist_backend import PyTorchDistBackend

            self.backendFuncs = PyTorchDistBackend(comms_world_info, commsParams)
        else:
            logger.error("Unsopported NW stack! ")
            comms_utils.gracefulExit()

        self.backendFuncs.initialize_backend(
            comms_world_info.master_ip,
            comms_world_info.master_port,
            backend=commsParams.backend,
        )
        self.backendFuncs.sayHello()

        # set basic collective info
        (
            local_rank,
            global_rank,
            world_size,
            group,
            curDevice,
            curHwDevice,
        ) = comms_utils.get_rank_details(
            self.backendFuncs
        )  # Getting ranks from backednFuncs object, since we cannot use MPI (e.g.: TPU) to launch all the processes

        self.collectiveArgs.numIters = self.numIters
        self.collectiveArgs.numWarmupIters = self.numWarmupIters
        self.collectiveArgs.group = group
        self.collectiveArgs.device = curDevice
        self.collectiveArgs.world_size = world_size
        self.collectiveArgs.global_rank = global_rank
        self.collectiveArgs.backendFuncs = self.backendFuncs
        # FIXME: 0 is a common case, need this info from execution graph for more accurate replay
        self.collectiveArgs.srcOrDst = 0
        # FIXME: assuming it's always sum for reduce/allreduce operations
        self.collectiveArgs.op = self.backendFuncs.get_reduce_op("sum")
        # FIXME: alwasy perfom blocking comms; may study non-blocking in the future
        self.collectiveArgs.asyncOp = not self.is_blocking
        self.collectiveArgs.ipTensor = None
        self.collectiveArgs.opTensor = None
        self.collectiveArgs.quant_threshold = commsParams.quant_threshold

    def runBench(self, comms_world_info, commsParams):
        """Run the execution graph replay benchmark:
        1) Each rank reads and filters its execution graph
        2) Execute communication replay [Skip if on dry-run mode]
        3) report stats and performance (if not dry-run)
        """
        logger.info(
            f"[Rank-{comms_world_info.global_rank}] reading execution graph from {self.exgr_file}"
        )
        self.comm_size = comms_world_info.world_size
        self.global_rank = comms_world_info.global_rank

        self.readGraph()

        # only setup and perform collectives if not dry run mode
        if not self.is_dry_run:
            self.setBench(comms_world_info, commsParams)
            # start benchmark
            self.benchTime()
        elif comms_world_info.global_rank == 0:
            print(
                "+ Dry run mode...No replaying, Only Rank 0 read and analyze the execution graph..."
            )

        # rank 0 reports statistics
        if comms_world_info.global_rank == 0:
            self.reportBenchTime()


def main():
    mpi_env_params = comms_utils.read_mpi_env_vars()

    exgrBench = ExgrReplayBench()
    parser = argparse.ArgumentParser(
        description="PARAM Execution Graph Replay Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = exgrBench.readArgs(parser)
    exgrBench.setExgrFile(args, mpi_env_params)
    exgrBench.checkArgs(args)

    time.sleep(1)
    comms_world_info = comms_utils.comms_world_info_holder(
        args.master_ip, args.master_port, args.num_tpu_cores, mpi_env_params
    )
    commsParams = comms_utils.commsParamsHolderBase(args)
    exgrBench.initBench(args)
    exgrBench.runBench(comms_world_info, commsParams)


if __name__ == "__main__":
    main()
