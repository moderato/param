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
        print("-----")
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
        # print(node.name, node.id, [(type(i), i.shape if torch.is_tensor(i) else i, i.dtype if torch.is_tensor(i) else i) for i in inputs], type(node.outputs[0]))
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
            # parse_schema doesn't fit well
            # func_schema = torch._C.parse_schema(n.op_schema)
            # edge case 1: int[1] in aten::sum, int[2] in aten::conv2d (manually fixed for now)
            # edge case 2: * in aten::emtpy_strided (currently don't see anything bad)

            types = [item for item in n.op_schema.split(' ') if ',' not in item]
            types = [re.sub(r'\[[0-9]\]', '[]', t) for t in types] # e.g. int[2] -> int[]
            input_types = [item if 'Tensor' not in item else 'Tensor' for item in types[:-3]] # e.g. Tensor(float) -> Tensor
            input_types[0] = re.sub(r'^.*?\(', '', input_types[0]) # Strip the op name
            output_type = types[-1] if 'Tensor' not in types[-1] else 'Tensor' # e.g. Tensor(float) -> Tensor
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
        time_per_iter = 0.0
        for iters in range(self.numWarmupIters + self.numIters):
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            event_1.record()
            for _, node in self.sorted_nodes:
                self.run_op(node)
            event_2.record()
            torch.cuda.synchronize()
            if iters >= self.numWarmupIters:
                time_per_iter += event_1.elapsed_time(event_2) / self.numIters
            self.reset_registry()
        print("Execution time: {:.2f} ms".format(time_per_iter))


def main():
    parser = argparse.ArgumentParser(description="Execution Graph Replay")
    parser.add_argument(
        "-w", "--warmup", type=int, default=1, help="Number of warm up iterations."
    )
    parser.add_argument(
        "-i", "--iteration", type=int, default=5, help="Number of replay iterations."
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
            # num_classes = 1000
            # target = torch.randn(batch_size, num_classes).softmax(dim=1).cuda()
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        skip_first=3,
                        wait=1,
                        warmup=1,
                        active=10,
                        start_execution_graph=1,
                        stop_execution_graph=2),
                    on_trace_ready=trace_handler,
                    on_execution_graph_ready=execution_graph_handler) as p:
                for _ in range(15):
                    optimizer.zero_grad()
                    output = an(data)
                    print(output.shape, target.shape)
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
