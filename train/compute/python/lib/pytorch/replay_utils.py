# N.B. Exgr utils required. Integration to Pytorch WIP.
import torch
from exec_graph_utils import NodeType
import io, json


# TODO: Add all torch dtypes to here

TORCH_DTYPES_RNG = {
    "int8": (torch.int8, torch.ones),
    "half": (torch.half, torch.ones),
    "int": (torch.int, torch.ones),
    "long": (torch.int64, torch.ones),
    "long int": (torch.int64, torch.ones),
    "float": (torch.float32, torch.randn),
    "double": (torch.float64, torch.randn)
}


def is_tensor(n, ip):
    return isinstance(ip, int) and 'Tensor' in n.input_types[ip]


def is_op(node):
    return (node.type == NodeType.OPERATOR and (node.parent is not None and node.parent.type != NodeType.OPERATOR))


def is_lowest_level_aten(op):
    return op.name.startswith("aten::") and (
        not op.children or op.name == "aten::addmm"
    )


def has_backward_parent(op):
    if not op.parent or op.parent.id == op.id:
        return False
    if is_backward(op):
        return True
    return has_backward_parent(op.parent)


def is_backward(op):
    return "autograd::engine::evaluate_function: " in op.name or "Optimizer" in op.name


def is_backward_aten(op):
    return op.name.startswith("aten::") and \
            (not op.children or op.name == "aten::convolution_backward") and \
            has_backward_parent(op)


def trace_handler(prof):
    # print(prof.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")
    pass


def execution_graph_handler(output_file_name):
    print(f"pytroch execution graph output: {output_file_name}")
    found_root_node = False
    with io.open(output_file_name, 'r') as f:
        eg_graph = json.load(f)
        assert "nodes" in eg_graph
        nodes = eg_graph["nodes"]
        for n in nodes:
            assert "name" in n
            if "__ROOT_PROCESS__" in n["name"]:
                found_root_node = True

    assert found_root_node
