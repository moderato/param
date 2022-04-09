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
import gc
from replay_utils import *
from os import path
from collections import defaultdict

from pprint import pprint
import GPUtil

import torch
import comms_utils
from comms_utils import paramCommsBench, paramTimer, paramProfile
from exec_graph_utils import ExecutionGraph

logger = logging.getLogger(__name__)


class ExgrReplayBench(paramCommsBench):
    def __init__(self):
        super().__init__(supportedNwstacks=["pytorch-dist", "pytorch-xla-tpu"])
        self.exgr = {}
        self.exgr_file = ""
        self.is_dry_run = False
        self.is_blocking = True
        self.do_warm_up = True
        self.allowList = ""
        self.tensor_registry = []

        self.strToTorchDtype = {
            "Byte": torch.uint8,
            "Float": torch.float32,
            "Int": torch.int32,
            "Long": torch.long,
            "Double": torch.double,
        }

    def run_op(self, node):
        # print("-----")
        # print(node.name, node.inputs, node.input_shapes)
        inputs = [
            self.tensor_registry[item] if is_tensor(node, idx) else \
            (
                None if item == '<None>' else \
                (
                    tuple(item) if isinstance(item, list) else item
                )
            ) for idx, item in enumerate(node.inputs)
        ]
        # print(node.name, [(type(i), i.shape if torch.is_tensor(i) else i) for i in inputs])
        output_id = node.outputs[0]
        func = self.funcs[node.id]
        output = func(*inputs)
        # print("Dependency count")
        # pprint(self.dependency)
        for idx, input_id in enumerate(node.inputs):
            # Only consider tensor id
            if not is_tensor(node, idx):
                continue
            # print(input_id, self.dependency[input_id])
            self.dependency[input_id] -= 1
            # print(input_id, self.dependency[input_id])
            if self.dependency[input_id] == 0:
                # print("delete tensor {}".format(input_id))
                del self.tensor_registry[input_id]
        self.tensor_registry[output_id] = output
        # GPUtil.showUtilization()
        # print("Tensor registry")
        # print(self.tensor_registry.keys())


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

    def reset_registry(self):
        self.tensor_registry = {}
        self.op_registry = {}
        self.dependency = self.dependency_permanent.copy()
        self.tensor_registry = {k: v.cuda() for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()

    def preprocess_graph(self):
        self.dependency_permanent = defaultdict(int)
        self.tensor_registry_permanent = {}

        # Sort and filter nodes
        # edge case 1 (?): aten::addmm has a sub op aten::expand (i.e. not the lowest level) that operates a tensor needed in the future
        nodes = self.exgr.get_nodes(clean=True)
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
        # Method 1: Manually choose ops (same as the current perf model, might be good for python-based replay if we need it)
        # self.sorted_nodes = [
        #     (id, node) for id, node in sorted_nodes 
        #     if is_op(node) and not to_skip(node) and to_consider(node)
        # ] # Filter modules, kernels, etc
        ##############
        # Method 2: aten in BW and op in FW (works for now)
        # self.sorted_nodes = [
        #     (id, node) for id, node in sorted_nodes 
        #     if (is_backward_aten(node)) or
        #         (is_op(node) and not is_backward(node)) # and to_consider(node))
        # ] # Filter modules, kernels, etc
        ##############
        # Method 3: lowest aten for both BW and FW (also works)
        self.sorted_nodes = [(id, node) for id, node in sorted_nodes if (is_lowest_level_aten(node))] # Filter modules, kernels, etc
        # pprint([(id, node.name) for id, node in self.sorted_nodes])

        # Tensors dependency
        for tid in self.exgr.tensors:
            for _, n in self.sorted_nodes:
                for idx, ip in enumerate(n.inputs):
                    if tid == ip and is_tensor(n, idx):
                        self.dependency_permanent[tid] += 1
        
        # Mark all intermediate tensors
        intermediate = set()
        input_set = set()
        for _, n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if is_tensor(n, idx) and ip in self.dependency_permanent.keys():
                    input_set.add(ip)

            # Tensors occurred as inputs before are not to be removed
            for o in n.outputs:
                if o in self.dependency_permanent and \
                        o not in input_set:
                    intermediate.add(o)

        # Instantiation of tensors:
        # edge case 1: one tensor id is reused multiple times, and each time has a different shape (taking the first shape should work?)
        for _, n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if is_tensor(n, idx) and \
                        ip not in self.tensor_registry_permanent.keys() and \
                        ip in self.dependency_permanent.keys() and \
                        ip not in intermediate: # Only take the first size
                    self.tensor_registry_permanent[ip] = torch.randn(n.input_shapes[idx], requires_grad=True)

        # Build aten funcs
        self.funcs = {}
        for _, n in self.sorted_nodes:
            input_count = len(n.input_types)
            # parse_schema doesn't fit well
            # func_schema = torch._C.parse_schema(n.op_schema)
            # edge case 1: int[1] in aten::sum (manually fixed for now)
            # edge case 2: * in aten::emtpy_strided (currently don't see anything bad)

            types = [item for item in n.op_schema.split(' ') if ',' not in item]
            input_types = [item if 'Tensor' not in item else 'Tensor' for item in types[:-3]]
            input_types[0] = re.sub(r'^.*?\(', '', input_types[0]) # Strip the op name
            output_type = types[-1] if 'Tensor' not in types[-1] else 'Tensor'
            output_type = output_type.lstrip('(').rstrip(')')

            inputStr = """
                graph({}):
                    %{}: {} = {}({})
                    return (%{})
            """.format(
                ", ".join(["%{}: {}".format(idx, t) for idx, t in enumerate(input_types)]),
                input_count + 1,
                output_type,
                n.name,
                ", ".join(["%{}".format(idx) for idx in range(input_count)]),
                input_count + 1,
            )

            # print(inputStr)
            # print("=============")
            graph = torch._C.parse_ir(inputStr)
            cu = torch._C.CompilationUnit()
            func = cu.create_function(n.name, graph)
            self.funcs[n.id] = func

        # Reset
        self.reset_registry()


    def benchTime(self):
        self.preprocess_graph()
        time_per_iter = 0

        # print(self.dependency_permanent)
        for iters in range(self.numWarmupIters + self.numIters):
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            event_1.record()
            for _, node in self.sorted_nodes:
                # TODO: Fix communication ops
                # if node.name in "record_param_comms":
                #     op = None
                #     # Get the collective node
                #     def dfs(n):
                #         nonlocal op
                #         if n.name in COMMS:
                #             op = n
                #         for c in n.children:
                #             dfs(c)
                #     dfs(node)
                #     if op is not None:
                #         self.prepComms(op)
                #         latency, _ = self.runComms(op.name.split("nccl:")[-1])
                #         time_per_iter += latency / 1e3 # In ms
                self.run_op(node)
            event_2.record()
            torch.cuda.synchronize()
            if iters >= self.numWarmupIters:
                time_per_iter += event_1.elapsed_time(event_2) / self.numIters
            # print("=============")
            self.reset_registry()

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
    mpi_env_params = comms_utils.read_comms_env_vars()

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
