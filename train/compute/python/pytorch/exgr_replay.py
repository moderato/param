import argparse
import json
import logging
import os
import sys
import shutil
import gc
import re
from collections import defaultdict

# N.B. Exgr utils required. Integration to Pytorch WIP.
from exec_graph_utils import ExecutionGraph

import torch
from ..lib import pytorch as lib_pytorch
from ..lib.init_helper import init_logging, load_modules
from ..lib.pytorch.replay_utils import *
from ..workloads import pytorch as workloads_pytorch
from ..workloads.pytorch.alex_net import AlexNet


class ExgrReplayManager:
    def __init__(self, exgr, args):
        with open(exgr, 'r') as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.numWarmupIters = args.warmup
        self.numIters = args.iteration

    def run_op(self, node):
        # print(node.name, node.inputs, node.input_shapes)
        inputs = [
            self.tensor_registry[item] if is_tensor(node, idx) else \
            (
                None if item == '<None>' else item
            ) for idx, item in enumerate(node.inputs)
        ]
        # print(node.name, node.id, [(type(i), i.shape if torch.is_tensor(i) else i, i.dtype if torch.is_tensor(i) else i) for i in inputs], type(node.outputs[0]))
        func, output_count = self.funcs[node.id]
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        if output_count == 1:
            outputs = (func(*inputs),)
        else:
            outputs = func(*inputs)
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
        for output_id, output in zip(node.outputs, outputs):
            self.tensor_registry[output_id] = output
        # print("Tensor registry")
        # print(self.tensor_registry.keys())

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
        nodes = self.exgr.get_nodes(clean=True)
        sorted_nodes = sorted(nodes.items(), key=lambda x: x[0])
        # Filter modules, kernels, etc
        # FW: Pytorch level ops (to prevent cases like aten::addmm having a sub op aten::expand (i.e. not the lowest level) that operates a tensor needed in the future)
        # BW: All lowest aten ops
        self.sorted_nodes = [
            (id, node) for id, node in sorted_nodes 
            if (is_backward_aten(node)) or
                (is_op(node) and not is_backward(node))
        ]
        # from pprint import pprint
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
        for _, n in self.sorted_nodes:
            for idx, ip in enumerate(n.inputs):
                if is_tensor(n, idx) and \
                        ip not in self.tensor_registry_permanent.keys() and \
                        ip in self.dependency_permanent.keys() and \
                        ip not in intermediate: # Only take the first size
                    dtype, rng = TORCH_DTYPES_RNG[n.input_types[idx].lstrip('Tensor(').rstrip(')')]
                    self.tensor_registry_permanent[ip] = \
                        rng(n.input_shapes[idx], requires_grad=True).to(dtype)

        # Build aten funcs
        self.funcs = {}
        for _, n in self.sorted_nodes:
            input_count = len(n.input_types)
            output_count = len(n.output_types)

            tmp = n.op_schema.split('->')
            types = [item for item in tmp[0].split(' ') if ',' not in item]
            types = [re.sub(r'\[[0-9]\]', '[]', t) for t in types][:-2] # e.g. int[2] -> int[]
            input_types = [t if 'Tensor' not in t else 'Tensor' for t in types] # e.g. Tensor(float) -> Tensor
            input_types[0] = re.sub(r'^.*?\(', '', input_types[0]) # Strip the op name
            output_types = tmp[-1].lstrip(' (').rstrip(')').split(', ')
            output_types = [t if 'Tensor' not in t else 'Tensor' for t in output_types]

            inputStr = """
                graph({}):
                    {} = {}({})
                    {}
                    return (%output)
            """.format(
                ", ".join(["%{}: {}".format(idx, t) for idx, t in enumerate(input_types)]),
                "%output: {}".format(output_types[0]) if output_count == 1 else ", ".join(["%{}: {}".format(idx + input_count, t) for idx, t in enumerate(output_types)]),
                n.name,
                ", ".join(["%{}".format(idx) for idx in range(input_count)]),
                "%output : ({}) = prim::TupleConstruct({})".format(
                    ", ".join(["Tensor" for _ in range(output_count)]),
                    ", ".join(["%{}".format(idx + input_count) for idx in range(output_count)])
                ) if output_count > 1 else "",
            )
            # print(inputStr)
            # print("=============")
            graph = torch._C.parse_ir(inputStr)
            cu = torch._C.CompilationUnit()
            func = cu.create_function(n.name, graph)
            self.funcs[n.id] = (func, output_count)

        # Reset
        self.reset_registry()


    def benchTime(self):
        self.preprocess_graph()
        total_time = 0.0
        for iter in range(self.numWarmupIters + self.numIters):
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            event_1.record()
            for _, node in self.sorted_nodes:
                self.run_op(node)
            event_2.record()
            torch.cuda.synchronize()
            if iter >= self.numWarmupIters:
                total_time += event_1.elapsed_time(event_2)
            self.reset_registry()
        print("Execution time: {:.2f} ms".format(total_time / self.numIters))


def main():
    parser = argparse.ArgumentParser(description="Execution Graph Replay")
    parser.add_argument(
        "-w", "--warmup", type=int, default=5, help="Number of warm up iterations."
    )
    parser.add_argument(
        "-i", "--iteration", type=int, default=30, help="Number of replay iterations."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="alex_net",
        help="File name prefix to write benchmark results.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        default="benchmark_result",
        help="File name prefix to write benchmark results.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase log output verbosity."
    )

    args = parser.parse_args()

    if args.verbose:
        logger = init_logging(logging.DEBUG)
    else:
        logger = init_logging(logging.INFO)

    # Load PyTorch implementations for data generator and operators.
    load_modules(lib_pytorch)

    # Load PyTorch operator workloads.
    load_modules(workloads_pytorch)

    exgr_json_path = "examples/pytorch/exgr_jsons/{}.json".format(args.model)
    if not os.path.exists(exgr_json_path):
        if args.model == "alex_net": # Default
            batch_size = 2
            an = AlexNet().cuda() # AlexNet
            data = torch.randn([batch_size, 3, 224, 224]).cuda() # NCHW
            optimizer = torch.optim.SGD(an.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss().cuda()
            target = torch.arange(1, batch_size + 1).long().cuda()
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            total_time = 0.0

            # Warmup
            for _ in range(5):
                optimizer.zero_grad()
                output = an(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Benchmark
            for _ in range(100):
                event_1.record()
                optimizer.zero_grad()
                output = an(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                event_2.record()
                torch.cuda.synchronize()
                total_time += event_1.elapsed_time(event_2) # In ms
            print("Time per iteration: {} ms".format(total_time / 100))

            # Collect exgr
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        skip_first=3,
                        wait=1,
                        warmup=1,
                        active=5,
                        start_execution_graph=1,
                        stop_execution_graph=2),
                    # on_trace_ready=trace_handler,
                    on_execution_graph_ready=execution_graph_handler) as p:
                for _ in range(10):
                    optimizer.zero_grad()
                    output = an(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    p.step()
                exgr_output = torch.profiler.get_execution_graph_observer_output_file_name()
                logger.info("Copy to exgr json to {}".format(exgr_json_path))
                shutil.copy(exgr_output, exgr_json_path)
        else:
            sys.error("Execution graph json file doesn't exist! Quit replay...")

    replay_manager = ExgrReplayManager(exgr_json_path, args)
    replay_manager.benchTime()

if __name__ == "__main__":
    main()
