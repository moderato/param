import argparse
import json
import logging
import os
import sys
import shutil
import gc
from collections import defaultdict

# N.B. Exgr utils required. Integration to Pytorch WIP.
from ..lib.pytorch.exec_graph_utils import ExecutionGraph

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

        # Permanent
        self.root_node_name = args.subgraph
        self.tensor_registry_permanent = {}
        self.dependency_permanent = defaultdict(int)
        self.sorted_nodes = []
        self.funcs = {}

        # Temporary
        self.tensor_registry = {}
        self.dependency = {}


    def reset_registry(self):
        self.dependency = self.dependency_permanent.copy()
        self.tensor_registry = {k: (v.cuda() if v is not None else None) for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()


    def get_subgraph(self):
        """
            return: root node of the subgraph
        """
        nodes = self.exgr.get_nodes(clean=True)
        assert isinstance(self.root_node_name, str)
        if self.root_node_name != "":
            # Look up root nodes of subgraph by name
            for _, n in nodes.items():
                if n.name == self.root_node_name:
                    return n
        return nodes[1] # 1-base


    def _dfs_traverse(self, node):
        # moc: maximum output count
        # peculiar: has more outputs than its parents
        # processed: the whole subtree has been processed (ops taken or neglected)
        def get_moc_and_peculiar(node, last_moc):
            if len(node.outputs) > last_moc:
                return len(node.outputs), True
            return last_moc, False

        def dfs(node, depth, moc, peculiar, processed):
            # Update info of the current node
            moc, peculiar = get_moc_and_peculiar(node, moc)
            ret_peculiar, ret_processed = peculiar, processed
            c_peculiars, c_processeds = [], []

            # Search the subtree
            for child in node.children:
                c_peculiar, c_processed = dfs(child, depth+1, moc, peculiar, processed)
                c_peculiars.append(c_peculiar)
                c_processeds.append(c_processed)

            next_level_has_peculiar = any(c_peculiars)
            next_level_has_processed = any(c_processeds)
            # Either there's a peculiar op or there's a processed subtree in the next level
            # Example: aten::cross_entropy_loss
            if next_level_has_peculiar or next_level_has_processed:
                for idxc, c in enumerate(node.children):
                    # Take this op if not processed
                    if not c_processeds[idxc] and is_qualified(c):
                        self.sorted_nodes.append((c.id, c))

                        # Tensors dependency
                        for idxi, ip in enumerate(c.inputs):
                            if is_tensor(c, idxi):
                                self.dependency_permanent[ip] += 1

                        # Build aten funcs
                        func, output_count = build_torchscript_func(c)
                        self.funcs[c.id] = (func, output_count)

                        # Mark as processed
                        c_processeds[idxc] = True

            # Mark an op and its subtree as processed if all branches are processed
            all_next_level_processed = len(c_processeds) != 0 and all(c_processeds)
            if all_next_level_processed:
                ret_processed = True

            return ret_peculiar, ret_processed

        dfs(node, 0, len(node.outputs), False, False)
        self.sorted_nodes = [(id, node) for id, node in sorted(self.sorted_nodes, key=lambda x: x[0])]


    def allocate_tensors(self):
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
                    try:
                        dtype, rng = TORCH_DTYPES_RNG[n.input_types[idx].lstrip('Tensor(').rstrip(')')]
                        self.tensor_registry_permanent[ip] = rng(n.input_shapes[idx]).to(dtype)
                    except KeyError:
                        self.tensor_registry_permanent[ip] = None
                    # except:
                    #     print(n.name, n.id, ip, n.input_shapes[idx])


    def preprocess_graph(self):
        # Get the subgraph
        root_node = self.get_subgraph()
        self._dfs_traverse(root_node)

        # Allocate
        self.allocate_tensors()

        # Reset
        self.reset_registry()


    def run_op(self, node):
        # print("-----")
        # print(node.name, node.inputs, node.outputs)
        inputs = [
            self.tensor_registry[item] if is_tensor(node, idx) else \
            (
                None if item == '<None>' else item
            ) for idx, item in enumerate(node.inputs)
        ]

        ######
        # Workaround to eliminate the "strides() called on undefined Tensor" error
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        ######

        # print(node.name, node.id, [(type(i), i.shape if torch.is_tensor(i) else i, i.dtype if torch.is_tensor(i) else i) for i in inputs], type(node.outputs[0]))
        func, output_count = self.funcs[node.id]
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
            if input_id not in node.outputs:
                self.dependency[input_id] -= 1
            # print(input_id, self.dependency[input_id])
            if self.dependency[input_id] == 0:
                # print("delete tensor {}".format(input_id))
                del self.tensor_registry[input_id]
                del self.dependency[input_id]
        for output_id, output in zip(node.outputs, outputs):
            self.tensor_registry[output_id] = output
        # print("Tensor registry (count: {})".format(len(self.tensor_registry.keys())))
        # pprint(self.tensor_registry.keys())
        # print("Tensor dependency")
        # pprint(self.dependency)


    def benchTime(self):
        self.preprocess_graph()
        total_time = 0.0
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ],
        # on_trace_ready=another_trace_handler) as p:
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
        "-g",
        "--subgraph",
        type=str,
        default="",
        help="Subgraph tag name.",
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
            for _ in range(10):
                optimizer.zero_grad()
                output = an(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # Benchmark
            # with torch.profiler.profile(
            #     activities=[
            #         torch.profiler.ProfilerActivity.CPU,
            #         torch.profiler.ProfilerActivity.CUDA,
            #     ],
            #     on_trace_ready=trace_handler) as p:
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
