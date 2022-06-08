import argparse
import json
import logging
import os
import tempfile
import sys
import shutil
import gc
from collections import defaultdict

import torch
from torch.profiler import ExecutionGraphObserver
from ..lib import pytorch as lib_pytorch
from ..lib.init_helper import init_logging, load_modules
from ..lib.pytorch.replay_utils import *
from ..workloads import pytorch as workloads_pytorch
from ..workloads.pytorch.alex_net import AlexNet

# N.B. Exgr utils required. Integration to Pytorch WIP.
from ..lib.pytorch.exec_graph_utils import ExecutionGraph


class ExgrReplayManager:
    def __init__(self, exgr, args):
        with open(exgr, 'r') as f:
            self.exgr = ExecutionGraph(json.load(f))
        self.numWarmupIters = args.warmup
        self.numIters = args.iteration
        self.profile_replay = args.profile_replay

        # Permanent
        self.root_node_name = args.subgraph
        self.skip_root_node_names = args.skip_subgraphs.split('-')
        self.tensor_registry_permanent = {}
        self.dependency_permanent = defaultdict(int)
        self.sorted_nodes = []
        self.funcs = {}
        self.verbose = args.verbose

        # Temporary
        self.tensor_registry = {}
        self.dependency = {}


    def reset_registry(self):
        self.dependency = self.dependency_permanent.copy()
        self.tensor_registry = {k: (v.cuda() if v is not None else None) for k, v in self.tensor_registry_permanent.items()}
        gc.collect()
        torch.cuda.empty_cache()


    def is_tensor_registered(self, t_id):
        return t_id in self.dependency_permanent.keys()


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


    def extract_subgraph(self, root):
        """
            return: all nodes in the subgraph, in the order of node ID
        """
        def _dfs_traverse(root):
            for child in root.children:
                if child.name in self.skip_root_node_names:
                    continue
                if (is_backward_aten(child)) or (is_op(child, strict=True) and not is_backward_parent(child)):
                    self.sorted_nodes.append(child)

                    # Tensors dependency
                    ip_set = set() # Prevent identical inputs to be marked multiple times
                    for _, ip, _ in child.get_input_tensors():
                        if ip not in ip_set:
                            self.dependency_permanent[ip] += 1
                            ip_set.add(ip)

                    # Build aten funcs
                    func, output_count = build_torchscript_func(child)
                    self.funcs[child.id] = (func, output_count)
                else:
                    _dfs_traverse(child)

        _dfs_traverse(root)
        self.sorted_nodes = [node for node in sorted(self.sorted_nodes, key=lambda x: x.id)]
        if self.verbose:
            logger.info("Sorted nodes:")
            pprint([(n.id, n.name) for n in self.sorted_nodes])

            logger.info("Tensor dependency:")
            pprint(self.dependency_permanent)


    def allocate_tensors(self):
        # Mark all intermediate tensors
        intermediate = set()
        input_set = set()
        for n in self.sorted_nodes:
            for _, t_id, _ in n.get_input_tensors():
                if self.is_tensor_registered(t_id):
                    input_set.add(t_id)

            # Tensors occurred as inputs before are not to be removed
            for _, t_id, _ in n.get_output_tensors():
                if self.is_tensor_registered(t_id) and t_id not in input_set:
                    intermediate.add(t_id)

        # Mark those tensors that occur first as an input to be instantiated later
        instantiate = set()
        output_set = set()
        for n in self.sorted_nodes:
            for (_, t_id, _) in n.get_input_tensors():
                if self.is_tensor_registered(t_id) and t_id not in output_set:
                    instantiate.add(t_id)

            for (_, t_id, _) in n.get_output_tensors():
                if self.is_tensor_registered(t_id):
                    output_set.add(t_id)

        # Instantiation of tensors:
        for n in self.sorted_nodes:
            for tp, t_id, shape in n.get_input_tensors():
                # if t_id == (142, 135, 36, 36, 8):
                #     print(n.name)
                #     print(self.is_tensor_registered(t_id))
                #     print(t_id not in self.tensor_registry_permanent.keys())
                #     print(t_id in instantiate)
                if self.is_tensor_registered(t_id) and \
                        t_id not in self.tensor_registry_permanent.keys() and \
                        t_id in instantiate: # Only take the first size
                    try:
                        dtype, rng = TORCH_DTYPES_RNG[tp.lstrip('Tensor(').rstrip(')')]
                        self.tensor_registry_permanent[t_id] = rng(shape).to(dtype)
                    except KeyError:
                        self.tensor_registry_permanent[t_id] = None
                    # except:
                    #     print(n.name, n.id, t_id, shape)


    def preprocess_graph(self):
        # Get the subgraph
        root_node = self.get_subgraph()
        self.extract_subgraph(root_node)

        # Allocate
        self.allocate_tensors()


    def run_op(self, node):
        # print("-----")
        # print(node.name, node.id, node.inputs, node.outputs)
        inputs = [
            self.tensor_registry[tuple(item)] if is_tensor(node, idx) else \
            (
                [self.tensor_registry[tuple(id)] for id in item] if is_tensor_list(node, idx) else \
                (
                    None if item == '<None>' else item
                )
            ) for idx, item in enumerate(node.inputs)
        ]

        ######
        # Workaround to eliminate the "strides() called on undefined Tensor" error
        if node.name == "aten::convolution_backward":
            inputs[-1] = [True, True, True]
        ######

        # print(node.name, node.id, [(type(i), i.shape if torch.is_tensor(i) else i, i.dtype if torch.is_tensor(i) else i) for i in inputs])
        func, output_count = self.funcs[node.id]
        if output_count == 1:
            tmp = (func(*inputs),)
        else:
            tmp = func(*inputs)
        # Flatten any tensor lists
        # TODO: Simplify this
        outputs = []
        for x in tmp:
            if isinstance(x, list) and isinstance(x[0], torch.Tensor):
                outputs.extend(x)
            elif isinstance(x, torch.Tensor):
                outputs.append(x)

        # print("Dependency count")
        # pprint(self.dependency)
        for tp, t_id, _ in node.get_input_tensors():
            # Only consider tensor id
            if 'Tensor' not in tp:
                continue
            # print(t_id, self.dependency[t_id])
            if t_id not in node.outputs:
                self.dependency[t_id] -= 1
            # print(t_id, self.dependency[t_id])
            if self.dependency[t_id] == 0:
                # print("delete tensor {}".format(t_id))
                del self.tensor_registry[t_id]
                del self.dependency[t_id]
        for (_, t_id, _), output in zip(node.get_output_tensors(), outputs):
            self.tensor_registry[t_id] = output
        # print("Tensor registry (count: {})".format(len(self.tensor_registry.keys())))
        # pprint(self.tensor_registry.keys())
        # print("Tensor dependency")
        # pprint(self.dependency)


    def benchTime(self):
        self.preprocess_graph()
        total_time = 0.0
        event_1 = torch.cuda.Event(enable_timing=True)
        event_2 = torch.cuda.Event(enable_timing=True)
        # N.B.: Use torch.autograd.profiler.profile instead of torch.profiler.profile for
        #       enabling/disabling replay profiling.
        with torch.autograd.profiler.profile(
            self.profile_replay, use_cuda=True, use_kineto=True, record_shapes=False
        ) as prof:
            for iter in range(self.numWarmupIters + self.numIters):
                self.reset_registry()
                event_1.record()
                for node in self.sorted_nodes:
                    self.run_op(node)
                event_2.record()
                torch.cuda.synchronize()
                if iter >= self.numWarmupIters:
                    total_time += event_1.elapsed_time(event_2)
            print("{} replay time{}: {:.2f} ms".format(
                make_subgraph_text(self.root_node_name),
                " (profiled)" if self.profile_replay else "",
                total_time / self.numIters
            ))
        if self.profile_replay:
            another_trace_handler(subgraph=self.root_node_name)(prof)


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
        "-k",
        "--skip-subgraphs",
        type=str,
        default="",
        help="Tag names of subgraphs to be skipped",
    )
    parser.add_argument(
        "-p", "--profile-replay", action="store_true", help="Profile replay and get trace."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase log output verbosity."
    )

    args = parser.parse_args()

    global logger
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
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            eg = ExecutionGraphObserver()
            eg.register_callback(fp.name)
            eg.start()
            with torch.autograd.profiler.record_function("module::ZeroGrad"):
                optimizer.zero_grad()
            with torch.autograd.profiler.record_function("module::Forward"):
                output = an(data)
            with torch.autograd.profiler.record_function("module::Backward_WeightsUpdate"):
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            eg.stop()
            eg.unregister_callback()
            logger.info("Copy to exgr json to {}".format(exgr_json_path))
            shutil.copy(fp.name, exgr_json_path)
        else:
            sys.error("Execution graph json file doesn't exist! Quit replay...")

    replay_manager = ExgrReplayManager(exgr_json_path, args)
    replay_manager.benchTime()

if __name__ == "__main__":
    main()
